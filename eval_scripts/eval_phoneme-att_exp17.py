import torch
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Processor, Wav2Vec2ForCTC
import json
from collections import defaultdict
from torch.nn.functional import log_softmax
from os.path import join
from datasets import load_metric, load_from_disk, Dataset, DatasetDict
import pandas as pd
import sys, os
#dataset_dir = '/srv/scratch/z5173707/phonological/datasets/timit_phoneme/'
dataset_dir = sys.argv[1]#'/srv/scratch/z5173707/phonological/datasets/timit_phoneme/'
model_dir= sys.argv[2] #'fine_tune/best/'
output_dir = sys.argv[3]

os.makedirs(output_dir,exist_ok=True)
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

def GroupLabel(batch, mappers=mappers):
    def mapToken(phList, mappers=mappers):
        g_labels = []
        for mapper in mappers:
            g_label = []
            for p in phList.split():
                assert p in mapper, "{0} not in mapper".format(p)
                g_label.append(mapper[p])
            g_labels.append(' '.join(g_label))
        return g_labels
    batch["target_text"] = list(map(mapToken, batch["phoneme"]))
    return batch

def extract_all_phoneme(batch):
  all_phonemes = [p for l in batch['phoneme'] for p in l.split()]
  vocab = list(set(all_phonemes))
  return {'vocabs':vocab}

def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector + (mask + 1e-45).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)

def map_to_phoneme_results(batch, model=None, processor=None, group_ids=None, phoneme2att=None, tokenizer_phoneme=None):
    speech = batch['speech']
    model.to('cuda')
    input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values.to('cuda')
    with torch.no_grad():
        logits = model(input_values).logits.to('cuda')
    ngroups = len(group_ids)
    log_props_all_masked = []
    for i in range(ngroups):
        mask = torch.zeros(logits.size()[2], dtype = torch.bool).to('cuda')
        mask[0] = True
        mask[list(group_ids[i].keys())] = True
        mask.unsqueeze_(0).unsqueeze_(0)
        log_probs = masked_log_softmax(vector=logits, mask=mask, dim=-1).masked_fill(~mask,0)
        log_props_all_masked.append(log_probs)
    log_probs_cat = torch.stack(log_props_all_masked, dim=0).sum(dim=0)
    log_probs_phoneme = torch.matmul(phoneme2att,log_probs_cat.transpose(1,2)).transpose(1,2).type(torch.cuda.FloatTensor)
    pred_ids = torch.argmax(log_probs_phoneme,dim=-1)
    batch['pred_str'] = tokenizer_phoneme.batch_decode(pred_ids,spaces_between_special_tokens=True)[0]
    return batch

#model_dir = join('fine_tune','best')
processor = Wav2Vec2Processor.from_pretrained(model_dir)
model = Wav2Vec2ForCTC.from_pretrained(model_dir)
#Get group ids
group_ids = [sorted(processor.tokenizer.convert_tokens_to_ids(group)) for group in groups]
group_ids = [dict([(x[1],x[0]+1) for x in list(enumerate(g))]) for g in group_ids] 


data = load_from_disk(dataset_dir)
if isinstance(data, DatasetDict):
    isdict=True
else:
    isdict=False

#data = data.map(GroupLabel, batched=True, batch_size=8, num_proc=12, load_from_cache_file=False)
#if isdict:
#    vocab = data.map(extract_all_phoneme,batched=True, batch_size=-1,remove_columns=data['test'].column_names)
#    vocab_list = list(set(vocab['validation']['vocabs']) | set(vocab['test']['vocabs']))
#else:
#    vocab = data.map(extract_all_phoneme,batched=True, batch_size=-1,remove_columns=data.column_names)
#    vocab_list = vocab['vocabs']
#vocab_dict = {v: k+1 for k, v in enumerate(vocab_list)}
#vocab_dict['<pad>'] = 0
#with open(join(output_dir,'phoneme_vocab.json'), 'w') as vocab_file:
#    json.dump(vocab_dict, vocab_file)

tokenizer_phoneme = Wav2Vec2CTCTokenizer("phoneme_vocab.json", pad_token="<pad>", word_delimiter_token="")


phoneme_list = list(tokenizer_phoneme.get_vocab().keys())

p2att = torch.zeros((tokenizer_phoneme.vocab_size, processor.tokenizer.vocab_size)).type(torch.cuda.FloatTensor)
for p in phoneme_list:
    for mapper in mappers:
        if p == processor.tokenizer.pad_token:
            p2att[tokenizer_phoneme.convert_tokens_to_ids(p),processor.tokenizer.pad_token_id] = 1
        else:
            p2att[tokenizer_phoneme.convert_tokens_to_ids(p), processor.tokenizer.convert_tokens_to_ids(mapper[p])] = 1


results =  data.map(map_to_phoneme_results, batched=False, fn_kwargs={'processor':processor, 'model':model, 'group_ids':group_ids,'tokenizer_phoneme':tokenizer_phoneme, 'phoneme2att':p2att}, load_from_cache_file=False)
results.save_to_disk(join(output_dir,'results.db'))
per_metric = load_metric('wer')

with open(join(output_dir,'results.txt'),'w') as f:
    if isinstance(results, DatasetDict):
        for dataset in results:
            print("{} PER: {:.3f}".format(dataset, per_metric.compute(predictions=results[dataset]["pred_str"], references=results[dataset]["phoneme"])),file=f)
    else:
        print("{} PER: {:.3f}".format('test', per_metric.compute(predictions=results["pred_str"], references=results["phoneme"])),file=f)
