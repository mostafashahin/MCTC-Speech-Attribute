print('importing...')
from datasets import load_dataset, load_metric, ClassLabel, load_from_disk
import random
import pandas as pd
import re
from os.path import join
import json
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, Trainer, TrainingArguments
import numpy as np
import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import sys

dataset_dir = '/srv/scratch/z5173707/phonological/datasets/timit_phoneme_valid_core/' 
nproc=30
ngpus = torch.cuda.device_count()
print('Done importing...')
#cache_dir = '/g/data/wa66/Mostafa/.cache/'
cache_dir='/srv/scratch/z5173707/cache'

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
number_items_per_group = [len(g) for g in groups]

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


#Should run with batched=True
def prepare_dataset(batch, processor=None):
   # check that all files have the correct sampling rate
   def processPerGroup(item):
       #I did this because using just tokenizer(item) interpret "semivowel" to two tokens, semivowel and vowel
       #TODO use unique name and not part of each other
       labels = processor.tokenizer([t.split() for t in item], is_split_into_words=True).input_ids
       #labels = [processor.tokenizer(t.split(),is_split_into_words=True) for t in item]
       return labels
   assert (
       len(set(batch["sampling_rate"])) == 1
   ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

   batch["input_values"] = processor.feature_extractor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values

   batch["labels"] = list(map(processPerGroup, batch["target_text"]))
   return batch

@dataclass
class DataCollatorMCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding_features: Union[bool, str] = True
    padding_labels: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding_features,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels_list = []
        nGroups = len(features[0]["labels"])
        for i in range(nGroups):
            label_features = [{"input_ids": feature["labels"][i]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(
                        label_features,
                        padding=self.padding_labels,
                        max_length=self.max_length_labels,
                        pad_to_multiple_of=self.pad_to_multiple_of_labels,
                        return_tensors="pt",
                    )
            labels_tmp = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100).unsqueeze(dim=1)
            labels_list.append(labels_tmp)

        batch["labels"] = torch.cat(labels_list,dim=1)

        return batch

#CTC loss using custom way
from torch import nn
import torch.nn.functional as F
class SCTCTrainer(Trainer):
    def __init__(self, **kargs):
        self.group_ids = kargs.pop('group_ids') #List with number of items in each group
        super(SCTCTrainer, self).__init__(**kargs)
        
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(inputs.get('input_values'))
        logits = outputs.get('logits')
        ngroups = len(self.group_ids)
        #first two tokens (0,1) for <pad> and <unk>
        assert labels.dim() == 3, "in multi-label 3D tensor is expected"
        assert ngroups == labels.size()[1], "Second dim should match number of groups"
        
        #IMPORTANT 0 reserved to <pad>  shared among all groups #VALIDATE THIS?!
        #IMPORTANT STARTING FROM 1, 1:1+n IS THE n ELEMENTS in FIRST GROUP and from 1+n:1+n+m IS THE M ELEMENTS IN SECOND GROUP
        #start_indx = 1 #0  for <pad>
        all_losses = []
        for i in range(ngroups):
            mask = torch.zeros(logits.size()[2], dtype = torch.bool)
            mask[0] = True
            mask[list(self.group_ids[i].keys())] = True


            targets = labels[:,i,:].squeeze()
            g_logits = logits[:,:,mask]
            log_probs = nn.functional.log_softmax(g_logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            #Label padding = -100
            labels_mask = targets >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = targets.masked_select(labels_mask)
            flattened_targets = flattened_targets.cpu().apply_(lambda x: self.group_ids[i][x]) #So all targets will start from index 1
            flattened_targets = flattened_targets.to(self.args.device)
            #flattened_targets = flattened_targets - start_indx +1 #So all targets will start from index 1
            #start_indx += self.number_items_per_group[i] 
            
            input_lengths = model._get_feat_extract_output_lengths(torch.ones_like(inputs.get('input_values'),dtype=torch.int32).sum(-1))
            loss = F.ctc_loss(log_probs, flattened_targets, input_lengths, target_lengths, blank=model.config.pad_token_id, zero_infinity=model.config.ctc_zero_infinity, reduction=model.config.ctc_loss_reduction)
            all_losses.append(loss)
        sctc_loss = sum(all_losses) #TODO: consider average over number of groups NOT VALID
        #TODO: consider reduction over input_lengths*target_lengths
        return (sctc_loss, outputs) if return_outputs else sctc_loss

def main():
    #Load timit dataset from prepared one should contain (speech, phoneme string)
    timit = load_from_disk(dataset_dir)


    #Get speech attribute labels
    timit = timit.map(GroupLabel, batched=True, batch_size=8, num_proc=nproc,load_from_cache_file=False)
    
    #NO <unk>
    #Create vocab from all elements in all groups
    #IMPORTANT 0 reserved to <pad> and shared among all groups
    vocab_list = np.unique(np.concatenate(groups)) 
    #Use one blank <pad> and one <unk> shared between all groups
    vocab_dict = {v: k+1 for k, v in enumerate(vocab_list)}
    vocab_dict['<pad>'] = 0
    vocab_dict = dict(sorted(vocab_dict.items(), key= lambda x: x[1]))
    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)
    print('vocab {0}'.format(' '.join(vocab_dict.keys())))



    #Build processor    
    tokenizer = Wav2Vec2CTCTokenizer("vocab.json", pad_token="<pad>", word_delimiter_token="")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


    #Get group ids dictionary
    group_ids = [sorted(tokenizer.convert_tokens_to_ids(group)) for group in groups]
    group_ids = [dict([(x[1],x[0]+1) for x in list(enumerate(g))]) for g in group_ids]

    #Get inputvalues and labels    
    timit_prepared = timit.map(prepare_dataset, remove_columns=timit['train'].column_names, batch_size=8, num_proc=nproc, batched=True,fn_kwargs={'processor':processor},keep_in_memory=True)

    timit_prepared.save_to_disk('timit_prep_3g')
    

    data_collator = DataCollatorMCTCWithPadding(processor=processor, padding_labels=True, max_length_labels=None)

    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-large-robust",
        gradient_checkpointing=True,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=processor.tokenizer.vocab_size,
        cache_dir=cache_dir
        ).to('cuda')
    
    model.freeze_feature_extractor()

    training_args = TrainingArguments(
      output_dir='fine_tune',
      #output_dir="./wav2vec2-base-timit-demo",
      group_by_length=True,
      per_device_train_batch_size=int(32/ngpus),
      evaluation_strategy="steps",
      fp16=True,
      num_train_epochs=30,
      save_steps=500,
      logging_steps=500,
      prediction_loss_only=True,
      learning_rate=1e-4,
      weight_decay=0.005,
      warmup_ratio=0.1,
      load_best_model_at_end=True,
      save_total_limit=2,
    )
    
    trainer = SCTCTrainer(
        model=model,
        #number_items_per_group=number_items_per_group,
        group_ids=group_ids,
        data_collator=data_collator,
        args=training_args,
        #compute_metrics=compute_metrics,
        train_dataset=timit_prepared['train'],
        eval_dataset=timit_prepared['test'],
        tokenizer=processor.feature_extractor,
    )
    
    trainer.train()
    
    model.save_pretrained(join('fine_tune','best'))
    processor.save_pretrained(join('fine_tune','best'))

if __name__=='__main__':
    main()
