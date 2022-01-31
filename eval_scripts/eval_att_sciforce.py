import torch
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Processor, Wav2Vec2ForCTC
from os.path import join
from datasets import load_from_disk, Dataset, DatasetDict
import pandas as pd
import sys
#dataset_dir = '/srv/scratch/z5173707/phonological/datasets/timit_phoneme/'

dataset_dir = sys.argv[1]#'/srv/scratch/z5173707/phonological/datasets/timit_phoneme/'
model_dir= sys.argv[2] #'fine_tune/best/'
output_dir = sys.argv[3]

#define groups
#make sure that all phonemes covered in each group
g1 = ['p_alveolar','n_alveolar']
g2 = ['p_anterior','n_anterior']
g3 = ['p_approximant','n_approximant']
g4 = ['p_bilabial','n_bilabial']
g5 = ['p_central','n_central']
g6 = ['p_close','n_close']
g7 = ['p_consonantal','n_consonantal']
g8 = ['p_continuant','n_continuant']
g9 = ['p_fricative','n_fricative']
g10 = ['p_front','n_front']
g11 = ['p_glottal','n_glottal']
g12 = ['p_labiodental','n_labiodental']
g13 = ['p_lateral','n_lateral']
g14 = ['p_mid','n_mid']
g15 = ['p_nasal','n_nasal']
g16 = ['p_nonsibfric','n_nonsibfric']
g17 = ['p_open','n_open']
g18 = ['p_palatal','n_palatal']
g19 = ['p_postalveolar','n_postalveolar']
g20 = ['p_round','n_round']
g21 = ['p_sibaff','n_sibaff']
g22 = ['p_sibfric','n_sibfric']
g23 = ['p_stop','n_stop']
g24 = ['p_tense','n_tense']
g25 = ['p_velar','n_velar']
g26 = ['p_voiced','n_voiced']
g27 = ['p_vowel','n_vowel']
g28 = ['p_sil','n_sil']
groups = [g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15,g16,g17,g18,g19,g20,g21,g22,g23,g24,g25,g26,g27,g28]
#Load attribute map
p_att = pd.read_csv('/srv/scratch/z5173707/phonological/sciforce_phone_att.csv',index_col=0)

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

def map_to_result(batch, processor=None, model=None, group_ids=None):
    input_values = processor(
          batch["speech"],
          sampling_rate=batch["sampling_rate"],
          return_tensors="pt").input_values
    
    if torch.cuda.is_available():
        model.to("cuda")
        input_values = input_values.to("cuda")    

    with torch.no_grad():
        logits = model(input_values).logits
    
    start_indx = 1
    pred = []
    for i in range(len(group_ids)):
        mask = torch.zeros(logits.size()[2], dtype = torch.bool)
        mask[0] = True
        mask[list(group_ids[i].values())] = True
        logits_g = logits[:,:,mask]
        pred_ids = torch.argmax(logits_g,dim=-1)
        #pred_ids[pred_ids>0] += start_indx - 1
        #start_indx += utils.number_items_per_group[i]
        pred_ids = pred_ids.cpu().apply_(lambda x: group_ids[i].get(x,x))
        pred.append(processor.batch_decode(pred_ids,spaces_between_special_tokens=True)[0])
    
    batch["pred_str"] = pred

    return batch

#Load timit test to speed up training
data = load_from_disk(dataset_dir)
data = data.map(GroupLabel, batched=True, batch_size=8, num_proc=12, load_from_cache_file=False)
#model_dir = join('fine_tune','best')
processor = Wav2Vec2Processor.from_pretrained(model_dir)
model = Wav2Vec2ForCTC.from_pretrained(model_dir)
#Get group ids
group_ids = [sorted(processor.tokenizer.convert_tokens_to_ids(group)) for group in groups]
group_ids = [dict([(x[0]+1,x[1]) for x in list(enumerate(g))]) for g in group_ids] #This is the inversion of the one used in training as here we need to map prediction back to original tokens
results =  data.map(map_to_result, batched=False, fn_kwargs={'processor':processor, 'model':model, 'group_ids':group_ids}, load_from_cache_file=False)
results.save_to_disk(join(output_dir,'results.db'))
from datasets import load_metric
ter = load_metric('wer')
ngroups = len(groups)
with open(join(output_dir,'results.txt'),'w') as f:
    if isinstance(results, DatasetDict):
        for dataset in results:
            for g in range(ngroups):
                pred = [item[g] for item in results[dataset]['pred_str']]
                target = [item[g] for item in results[dataset]['target_text']]
                print("{} group {} AER: {:.3f}".format(dataset,g,ter.compute(predictions=pred, references=target)),file=f)
    else:
        for g in range(ngroups):
            pred = [item[g] for item in results['pred_str']]
            target = [item[g] for item in results['target_text']]
            print("Test group {} AER: {:.3f}".format(g,ter.compute(predictions=pred, references=target)),file=f)

