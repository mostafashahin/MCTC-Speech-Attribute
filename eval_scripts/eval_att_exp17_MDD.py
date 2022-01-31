import sys
from datasets import load_from_disk
from os.path import join
from transformers import Wav2Vec2CTCTokenizer
import pandas as pd 
import json
import jiwer.transforms as tr
from typing import Union, List, Tuple, Dict
import Levenshtein
import numpy as np

nproc = 30

results_dir = sys.argv[1]
l2arctic_phoneme_map_file = '/srv/scratch/z5173707/Dataset/l2arctic/phoneme_map'
results = load_from_disk(join(results_dir,'results.db/'))

#define groups
#make sure that all phonemes covered in each group
g1 = ['p_alveolar','n_alveolar']
g2 = ['p_palatal','n_palatal']
g3 = ['p_dental','n_dental']
g4 = ['p_glottal','n_glottal']
g5 = ['p_labial','n_labial']
g6 = ['p_velar','n_velar']
g7 = ['p_anterior','n_anterior']
g8 = ['p_posterior','n_posterior']
g9 = ['p_retroflex','n_retroflex']
g10 = ['p_mid','n_mid']
g11 = ['p_high_v','n_high_v']
g12 = ['p_low','n_low']
g13 = ['p_front','n_front']
g14 = ['p_back','n_back']
g15 = ['p_central','n_central']
g16 = ['p_consonant','n_consonant']
g17 = ['p_sonorant','n_sonorant']
g18 = ['p_long','n_long']
g19 = ['p_short','n_short']
g20 = ['p_vowel','n_vowel']
g21 = ['p_semivowel','n_semivowel']
g22 = ['p_fricative','n_fricative']
g23 = ['p_nasal','n_nasal']
g24 = ['p_stop','n_stop']
g25 = ['p_approximant','n_approximant']
g26 = ['p_affricate','n_affricate']
g27 = ['p_liquid','n_liquid']
g28 = ['p_continuant','n_continuant']
g29 = ['p_monophthong','n_monophthong']
g30 = ['p_diphthong','n_diphthong']
g31 = ['p_round','n_round']
g32 = ['p_voiced','n_voiced']
g33 = ['p_bilabial','n_bilabial']
g34 = ['p_coronal','n_coronal']
g35 = ['p_dorsal','n_dorsal']
groups = [g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15,g16,g17,g18,g19,g20,g21,g22,g23,g24,g25,g26,g27,g28,g29,g30,g31,g32,g33,g34,g35]
#Load attribute map
p_att = pd.read_csv('/srv/scratch/z5173707/phonological/phonological_attributes_v12.csv',index_col=0)

#Create mapper for each group
mappers = []
for g in groups:
    p2att = {}
    for att in g:
        att_phs = p_att[p_att[att]==1].index
        for ph in att_phs:
            p2att[ph] = att
    mappers.append(p2att)

vocab_list = np.unique(np.concatenate(groups))
#Use one blank <pad> and one <unk> shared between all groups
vocab_dict = {v: k+1 for k, v in enumerate(vocab_list)}
vocab_dict['<pad>'] = 0
vocab_dict = dict(sorted(vocab_dict.items(), key= lambda x: x[1]))
with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)
print('vocab {0}'.format(' '.join(vocab_dict.keys())))
tokenizer = Wav2Vec2CTCTokenizer("vocab.json", pad_token="<pad>", word_delimiter_token="")

phone_map = pd.read_csv(l2arctic_phoneme_map_file,names=['symbol','phone'],delimiter='\t',keep_default_na=False)
ph_map_dict = dict(phone_map.values)

default_transform = tr.Compose(
    [
        tr.RemoveMultipleSpaces(),
        tr.Strip(),
        tr.SentencesToListOfWords(),
        tr.RemoveEmptyStrings(),
    ]
)

def preprocess(
    truth: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    truth_transform: Union[tr.Compose, tr.AbstractTransform],
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform],
    tokenizer: Wav2Vec2CTCTokenizer,
) -> Tuple[str, str]:
    """
    Pre-process the truth and hypothesis into a form that Levenshtein can handle.
    :param truth: the ground-truth sentence(s) as a string or list of strings
    :param hypothesis: the hypothesis sentence(s) as a string or list of strings
    :param truth_transform: the transformation to apply on the truths input
    :param hypothesis_transform: the transformation to apply on the hypothesis input
    :return: the preprocessed truth and hypothesis
    """

    # Apply transforms. By default, it collapses input to a list of words
    truth = truth_transform(truth)
    hypothesis = hypothesis_transform(hypothesis)

    # raise an error if the ground truth is empty
    if len(truth) == 0:
        raise ValueError("the ground truth cannot be an empty")
    offset = 0
    if tokenizer.unk_token_id:
        offset = 2
    else:
        offset = 1
    truth = [i-offset for i in tokenizer.convert_tokens_to_ids(truth)]
    hypothesis = [i-offset for i in tokenizer.convert_tokens_to_ids(hypothesis)]
    truth_chars = [chr(p) for p in truth]
    hypothesis_chars = [chr(p) for p in hypothesis]

    truth_str = "".join(truth_chars)
    hypothesis_str = "".join(hypothesis_chars)


    return truth_str, hypothesis_str

def compute_mdd_metrics(batch):
    def count_metrics(example,g):
        FA = FR = TR = TA = CD = DE = 0
        ref = example['target_text'][g]
        hypth = example['pred_str'][g]
        ori_indx = example['ori_indx']
        pron_errors = example['annotation']['phones']['error']
        del_errors = np.where(np.asarray(pron_errors)=='d')[0].shape[0]
        act_trans = example['annotation']['phones']['actual']
        exp_trans = example['annotation']['phones']['expected']
        nump_phones = len(ref.split())
        asr_error = [np.asarray(['hit']*nump_phones,dtype='U7'),np.arange(nump_phones),np.arange(nump_phones)]
        asr_ins_errors = []
        truth, hypothesis = preprocess(ref, hypth, default_transform, default_transform, tokenizer)
        ops = Levenshtein.editops(truth, hypothesis)
        for op in ops:
            pos = op[1]
            if op[0] == 'insert':
                asr_ins_errors.append(op)
                asr_error[2][pos:] += 1
            else:
                pos = op[1]
                asr_error[0][pos] = op[0]
                asr_error[1][pos] = op[1]
                asr_error[2][pos] = op[2]
                if op[0] == 'delete':
                    asr_error[2][pos+1:] -= 1
        #Check hit
        indxs = np.where(asr_error[0] == 'hit')[0]
        for i,j in zip(asr_error[1][indxs],asr_error[2][indxs]):
            assert ref.split()[i] == hypth.split()[j], '{} {} {}'.format(n,ref.split()[i],hypth.split()[j])
        asr_out = hypth.split()
        for asr_evl, ref_pos, asr_pos, ori_pos in zip(*asr_error,ori_indx):
            if asr_evl == 'hit':
                if pron_errors[ori_pos] == 'c':
                    TA += 1
                elif pron_errors[ori_pos] == 's':
                    exp_ph = ph_map_dict[exp_trans[ori_pos]]
                    exp_att = mappers[g][exp_ph]
                    act_att = ref.split()[ref_pos]
                    if exp_att == act_att:
                        TA += 1
                    else:
                        TR += 1
                        CD += 1
                elif pron_errors[ori_pos] == 'a':
                    CD += 1
                    TR += 1
                else:
                    print(asr_evl, ref_pos, asr_pos, ori_pos,pron_errors[ori_pos],act_trans[ori_pos],exp_trans[ori_pos],ref.split()[ref_pos],hypth.split()[asr_pos])
            elif asr_evl == 'replace':
                if pron_errors[ori_pos] == 'c':
                    FR += 1
                elif pron_errors[ori_pos] == 's':
                    exp_ph = ph_map_dict[exp_trans[ori_pos]]
                    exp_att = mappers[g][exp_ph]
                    act_att = ref.split()[ref_pos]
                    if exp_att == act_att:
                        FR += 1
                    else:
                        FA += 1
                elif pron_errors[ori_pos] == 'a':
                    TR += 1
                    DE += 1
                else:
                    print(asr_evl, ref_pos, asr_pos, ori_pos,pron_errors[ori_pos],act_trans[ori_pos],exp_trans[ori_pos],ref.split()[ref_pos],hypth.split()[asr_pos])
            elif asr_evl == 'delete':
                if pron_errors[ori_pos] == 'c':
                    FR += 1
                elif pron_errors[ori_pos] == 's':
                    exp_ph = ph_map_dict[exp_trans[ori_pos]]
                    exp_att = mappers[g][exp_ph]
                    act_att = ref.split()[ref_pos]
                    if exp_att == act_att:
                        FR += 1
                    else:
                        TR += 1
                        DE += 1
                elif pron_errors[ori_pos] == 'a':
                    FA += 1
                else:
                    print(asr_evl, ref_pos, asr_pos, ori_pos,pron_errors[ori_pos],act_trans[ori_pos],exp_trans[ori_pos],ref.split()[ref_pos],hypth.split()[asr_pos])
        for asr_eval, ref_pos, asr_pos in asr_ins_errors:
            if ref_pos >= len(ori_indx):
                FR += 1
            else:
                crt_ori_indx = ori_indx[ref_pos]
                if crt_ori_indx == 0:
                    FR += 1
                elif pron_errors[ori_indx[ref_pos]-1] == 'd':
                    del_errors -= 1
                    inserted_att = hypth.split()[asr_pos]
                    deleted_ph = ph_map_dict[exp_trans[ori_indx[ref_pos]-1]]
                    deleted_att = mappers[g][deleted_ph]
                    if inserted_att == deleted_att:
                        FA += 1
                    else:
                        TR += 1
                        DE += 1
                else:
                    FR += 1
        TR += del_errors
        CD += del_errors
        return((FA,FR,TR,TA,CD,DE))
    ngroups = len(batch['target_text'])
    batch['eval_MDD'] = [count_metrics(batch,g) for g in range(ngroups)]
    return(batch)


results = results.map(compute_mdd_metrics, num_proc=nproc)
results.save_to_disk(join(results_dir,'results_mdd.db'))

mdd_train = np.asarray(results['train']['eval_MDD'])
mdd_test = np.asarray(results['test']['eval_MDD'])



with open(join(results_dir,'results_mdd.txt'),'w') as f:
    mdd_train_g = mdd_train.sum(axis=0)
    FA,FR,TR,TA,CD,DE = mdd_train_g.transpose()
    FRR = FR/(TA+FR)
    FAR = FA/(FA+TR)
    DER = DE/(CD+DE)

    for g in range(len(groups)):
    	print('Train,{},{},{},{},{},{},{},{},{},{}'.format(g, FA[g],FR[g],TR[g],TA[g],CD[g],DE[g],FRR[g],FAR[g],DER[g]), file=f)
    mdd_test_g = mdd_test.sum(axis=0)
    FA,FR,TR,TA,CD,DE = mdd_test_g.transpose()
    FRR = FR/(TA+FR)
    FAR = FA/(FA+TR)
    DER = DE/(CD+DE)

    for g in range(len(groups)):
        print('Test,{},{},{},{},{},{},{},{},{},{}'.format(g, FA[g],FR[g],TR[g],TA[g],CD[g],DE[g],FRR[g],FAR[g],DER[g]), file=f)

