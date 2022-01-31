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

dataset_dir = '/srv/scratch/z5173707/phonological/datasets/timit_phoneme/'
nproc = 30
ngpus = torch.cuda.device_count()
print(ngpus)
cache_dir='/srv/scratch/z5173707/cache'
cer_metric = load_metric("cer")

#Should run with batched=True and batch_size <=0 or none (to process the whole data at once)
def extract_all_phoneme(batch):
  all_phonemes = [p for l in batch['phoneme'] for p in l.split()]
  vocab = list(set(all_phonemes))
  return {'vocabs':vocab}

#Should run with batched=True
def prepare_dataset(batch, processor=None):
    # check that all files have the correct sampling rate
    assert (
        len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["phoneme"]).input_ids
    return batch


@dataclass
class DataCollatorCTCWithPadding:
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
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

def make_compute_metric_fn(processor):
    def compute_metrics(pred,processor=processor):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
    
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    
        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    
        per = cer_metric.compute(predictions=pred_str, references=label_str)
    
        return {"wer": per}
    return compute_metrics

def main():
    #Load timit dataset from prepared one should contain (speech, phoneme string)
    timit = load_from_disk(dataset_dir)
    
    #Get Vocab
    vocab = timit.map(extract_all_phoneme, batched=True, batch_size=-1,remove_columns=timit['train'].column_names, keep_in_memory=True)
    #Using 61 phoneme-set
    vocab_list = list(set(vocab['train']['vocabs']) | set(vocab['test']['vocabs']))
    vocab_dict = {v: k+2 for k, v in enumerate(vocab_list)}
    vocab_dict['<pad>'] = 0
    vocab_dict['<unk>'] = 1
    #vocab_dict['|'] = len(vocab_dict)
    import json
    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    #Build processor

    tokenizer = Wav2Vec2CTCTokenizer("vocab.json", unk_token="<unk>", pad_token="<pad>", word_delimiter_token="")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
  
    #Data Prep 
    timit_prepared = timit.map(prepare_dataset, remove_columns=timit.column_names["train"], batch_size=8, num_proc=nproc, batched=True,fn_kwargs={'processor':processor},keep_in_memory=True)
 
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    
    #Training
    model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-robust",
    gradient_checkpointing=True,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=processor.tokenizer.vocab_size,
    )

    model.freeze_feature_extractor()


    output_dir = 'fine_tune'
    training_args = TrainingArguments(
    output_dir=output_dir,
    group_by_length=True,
    per_device_train_batch_size=int(32/ngpus),
    evaluation_strategy="steps",
    num_train_epochs=20,
    fp16=True,
    save_steps=500,
    eval_steps=500,
    logging_steps=500,
    learning_rate=1e-4,
    weight_decay=0.005,
    warmup_steps=1000,
    save_total_limit=2,
    disable_tqdm=False,
    #load_best_model_at_end=True,
    )


    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=timit_prepared["train"],
        eval_dataset=timit_prepared["test"],
        tokenizer=processor.feature_extractor,
    )
    trainer.compute_metrics = make_compute_metric_fn(processor)
    trainer.train()
    model.save_pretrained(join(output_dir,'best'))
    processor.save_pretrained(join(output_dir,'best'))

 
    
if __name__=='__main__':
    main()
 
