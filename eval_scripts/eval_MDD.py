import sys
from datasets import load_from_disk
from os.path import join
from transformers import Wav2Vec2CTCTokenizer
import pandas as pd
import jiwer.transforms as tr
from typing import Union, List, Tuple, Dict
import Levenshtein
import numpy as np

nproc = 30

results_dir = sys.argv[1]
l2arctic_phoneme_map_file = '/srv/scratch/z5173707/Dataset/l2arctic/phoneme_map' 
results = load_from_disk(join(results_dir,'results.db/'))
tokenizer_phoneme = Wav2Vec2CTCTokenizer("phoneme_vocab.json", pad_token="<pad>", word_delimiter_token="")

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
    def count_metrics(example):
        FA = FR = TR = TA = CD = DE = 0
        ref = example['phoneme']
        hypth = example['pred_str']
        ori_indx = example['ori_indx']
        pron_errors = example['annotation']['phones']['error']
        del_errors = np.where(np.asarray(pron_errors)=='d')[0].shape[0]
        act_trans = example['annotation']['phones']['actual']
        exp_trans = example['annotation']['phones']['expected']
        nump_phones = len(ref.split())
        asr_error = [np.asarray(['hit']*nump_phones,dtype='U7'),np.arange(nump_phones),np.arange(nump_phones)]
        asr_ins_errors = []
        truth, hypothesis = preprocess(ref, hypth, default_transform, default_transform, tokenizer_phoneme)
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
                    #if asr_out[asr_pos] == ph_map_dict[exp_trans[ori_pos]]:
                    #    TA += 1
                    #else:
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
                    if asr_out[asr_pos] == ph_map_dict[exp_trans[ori_pos]]:
                        FA += 1
                    else:
                        TR += 1
                        DE += 1
                elif pron_errors[ori_pos] == 'a':
                    TR += 1
                    DE += 1
                else:
                    print(asr_evl, ref_pos, asr_pos, ori_pos,pron_errors[ori_pos],act_trans[ori_pos],exp_trans[ori_pos],ref.split()[ref_pos],hypth.split()[asr_pos])
            elif asr_evl == 'delete':
                if pron_errors[ori_pos] == 'c':
                    FR += 1
                elif pron_errors[ori_pos] == 's':
                    TR += 1
                    DE += 1
                elif pron_errors[ori_pos] == 'a': #skip the case of not detecting insertion errors
                    FA += 1
                    #FA -= 1
                else:
                    print(asr_evl, ref_pos, asr_pos, ori_pos,pron_errors[ori_pos],act_trans[ori_pos],exp_trans[ori_pos],ref.split()[ref_pos],hypth.split()[asr_pos])
        for asr_eval, ref_pos, asr_pos in asr_ins_errors:
            if ref_pos >= len(ori_indx):
                FR += 1
            else:
                crt_ori_indx = ori_indx[ref_pos]
                
                if crt_ori_indx == 0:
                    FR += 1
                elif pron_errors[crt_ori_indx-1] == 'd':
                    del_errors -= 1
                    inserted_phoneme = hypth.split()[asr_pos]
                    deleted_phoneme = ph_map_dict[exp_trans[ori_indx[ref_pos]-1]]
                    if inserted_phoneme == deleted_phoneme:
                        FA += 1
                    else:
                        TR += 1
                        DE += 1
                    #print(inserted_phoneme, deleted_phoneme)
                    #print(ref_pos, asr_pos, hypth.split()[asr_pos], ref.split()[ref_pos], ori_indx[ref_pos], act_trans[ori_indx[ref_pos]], pron_errors[ori_indx[ref_pos]-1], exp_trans[ori_indx[ref_pos]-1], ref, hypth, act_trans, exp_trans, ori_indx, pron_errors, sep='\n')
                else:
                    FR += 1
        TR += del_errors
        CD += del_errors
        return((FA,FR,TR,TA,CD,DE))
    batch['eval_MDD'] = count_metrics(batch)
    return(batch)

results = results.map(compute_mdd_metrics, num_proc=nproc)
results.save_to_disk(join(results_dir,'results_mdd.db'))

mdd_train = np.asarray(results['train']['eval_MDD'])
mdd_test = np.asarray(results['test']['eval_MDD'])
#FA,FR,TR,TA,CD,DE = mdd.sum(axis=0)
#FRR = FR/(TA+FR)
#FAR = FA/(FA+TR)
#DER = DE/(CD+DE)

with open(join(results_dir,'results_mdd.txt'),'w') as f, open('MDD_results','a') as f_mdd:
    FA,FR,TR,TA,CD,DE = mdd_train.sum(axis=0)
    FRR = FR/(TA+FR)
    FAR = FA/(FA+TR)
    DER = DE/(CD+DE)
    print('Train results',file=f)
    print('FA = {}, FR = {}, TR = {}, TA = {}, CD = {}, DE = {}'.format(FA,FR,TR,TA,CD,DE), file=f)
    print('FRR = {}, FAR = {}, DER = {}'.format(FRR,FAR,DER), file=f)
    print(FA,FR,TR,TA,CD,DE,file=f_mdd)
    FA,FR,TR,TA,CD,DE = mdd_test.sum(axis=0)
    FRR = FR/(TA+FR)
    FAR = FA/(FA+TR)
    DER = DE/(CD+DE)
    print('Test results',file=f)
    print('FA = {}, FR = {}, TR = {}, TA = {}, CD = {}, DE = {}'.format(FA,FR,TR,TA,CD,DE), file=f)
    print('FRR = {}, FAR = {}, DER = {}'.format(FRR,FAR,DER), file=f)
    print(FA,FR,TR,TA,CD,DE,file=f_mdd)

