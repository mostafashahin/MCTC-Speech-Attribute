from datasets import load_from_disk
import pandas as pd
import librosa
import soundfile as sf

nproc=10
l2arctic = load_from_disk('/srv/scratch/z5173707/Dataset/l2arctic/data/l2arctic/')
suitcase = load_from_disk('/srv/scratch/z5173707/Dataset/l2arctic/data/suitcase')

#Get only annotated data
l2arctic_annotated = l2arctic.filter(lambda x:x['is_annotated'])
suitcase_annotated = suitcase.filter(lambda x:x['is_annotated'])

#Get phoneme from actual annotation
def get_phonemes(batch):
  batch['phoneme'] = batch['annotation']['phones']['actual']
  return batch
l2arctic_annotated = l2arctic_annotated.map(get_phonemes)
suitcase_annotated = suitcase_annotated.map(get_phonemes)

phone_map = pd.read_csv('/srv/scratch/z5173707/Dataset/l2arctic/phoneme_map',names=['symbol','phone'],delimiter='\t',keep_default_na=False)

def mapPhone(batch):
  def mapToken(phList):
    return [phone_map[phone_map.symbol==p].phone.values[0] if p in phone_map.symbol.values else p for p in phList]
  def RemoveEmptyTokens(phList):
    return [p for p in phList if p]
  batch['phoneme'] = list(map(mapToken, batch["phoneme"]))
  batch['phoneme'] = [' '.join(t) for t in map(RemoveEmptyTokens, batch["phoneme"])]
  return batch

suitcase_annotated = suitcase_annotated.map(mapPhone,batched=True)
l2arctic_annotated = l2arctic_annotated.map(mapPhone, batched=True)

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = sf.read(batch["file"])
    if sampling_rate != 16000:
        speech_array = librosa.resample(speech_array,sampling_rate,16000)
    batch["speech"] = speech_array
    batch["sampling_rate"] = 16000
    return batch

suitcase_annotated = suitcase_annotated.map(speech_file_to_array_fn,num_proc=nproc)
l2arctic_annotated = l2arctic_annotated.map(speech_file_to_array_fn,num_proc=nproc)

l2arctic_annotated.save_to_disk('/srv/scratch/z5173707/phonological/datasets/l2arctic/l2arctic')
suitcase_annotated.save_to_disk('/srv/scratch/z5173707/phonological/datasets/l2arctic/suitcase')
