from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_metric, load_dataset, load_from_disk, Dataset, DatasetDict
import re
from os.path import join
import soundfile as sf
import torch
per_metric = load_metric('wer')
nproc=30
import sys

dataset_dir = sys.argv[1]#'/srv/scratch/z5173707/phonological/datasets/timit_phoneme/'
model_dir= sys.argv[2] #'fine_tune/best/'
output_dir = sys.argv[3]

cache_dir='/srv/scratch/z5173707/cache'

def map_to_result(batch):
  model.to("cuda")
  input_values = processor(
      batch["speech"],
      sampling_rate=batch["sampling_rate"],
      return_tensors="pt"
  ).input_values.to("cuda")

  with torch.no_grad():
    logits = model(input_values).logits

  pred_ids = torch.argmax(logits, dim=-1)
  batch["pred_str"] = processor.batch_decode(pred_ids,spaces_between_special_tokens=True)[0]

  return batch


processor = Wav2Vec2Processor.from_pretrained(model_dir)
model = Wav2Vec2ForCTC.from_pretrained(model_dir)

#Load timit dataset from prepared one should contain (speech, phoneme string)
data = load_from_disk(dataset_dir)

results = data.map(map_to_result)
results.save_to_disk(join(output_dir,'results.db'))
with open(join(output_dir,'results.txt'),'w') as f:
	if isinstance(results, DatasetDict):
		for dataset in results:
			print("{} PER: {:.3f}".format(dataset, per_metric.compute(predictions=results[dataset]["pred_str"], references=results[dataset]["phoneme"])),file=f)
	else:
		print("{} PER: {:.3f}".format('test', per_metric.compute(predictions=results["pred_str"], references=results["phoneme"])),file=f)
      
