import sys
from datasets import load_dataset
import wave, struct
import numpy as np

cache_dir = '/srv/scratch/z5173707/cache'
save_dir = sys.argv[1]

p2p_red1 = {
    'ao':'aa',
    'ax':'ah',
    'ax-h':'ah',
    'axr':'er',
    'hv':'hh',
    'ix':'ih',
    'el':'l',
    'em':'m',
    'en':'n',
    'nx':'n',
    'eng':'ng',
    'dx':'d',
    'ux':'uw',
    'pcl':'',
    'tcl':'',
    'kcl':'',
    'bcl':'',
    'dcl':'',
    'gcl':'',
    'h#':'',
    'pau':'',
    'epi':'',
    'q':'',
}

def get_phonemes(batch):
  batch['file_path'] = batch['file']
  batch['phoneme'] = batch['phonetic_detail']['utterance']
  return batch


def mapPhone(batch):
  def mapToken(phList):
    return [p2p_red1[p] if p in p2p_red1 else p for p in phList]
  def RemoveEmptyTokens(phList):
    return [p for p in phList if p]
  batch['phoneme'] = list(map(mapToken, batch["phoneme"]))
  batch['phoneme'] = [' '.join(t) for t in map(RemoveEmptyTokens, batch["phoneme"])]
  return batch

#Should run with batched=False
def speech_file_to_array_fn(batch):
    with wave.open(batch["file_path"]+'.wav', 'rb') as s_f:
        sampling_rate = s_f.getframerate()
        data = s_f.readframes(s_f.getnframes())
        #data = list(struct.iter_unpack('h',data))
        #data = np.double(data)
        data = np.asarray([i[0] for i in struct.iter_unpack('h',data)], dtype=np.float64)
        speech_array = data / (2.0 ** 15)
    batch["speech"] = speech_array
    batch["sampling_rate"] = sampling_rate
    return batch


nproc=30
timit = load_dataset("timit_asr", cache_dir=cache_dir)
timit = timit.map(get_phonemes,remove_columns=timit['train'].column_names, batched=False)
timit = timit.map(mapPhone,batched=True, num_proc=nproc)
timit = timit.map(speech_file_to_array_fn, remove_columns=['file_path'], num_proc=nproc, load_from_cache_file=False)
timit.save_to_disk(save_dir)
