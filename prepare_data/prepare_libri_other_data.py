import sys
import librosa
import soundfile as sf
from datasets import load_dataset, DatasetDict
from dp.phonemizer import Phonemizer
import cmudict
import re
from os.path import join

version = sys.argv[1]
save_dir = sys.argv[2]
cache_dir='/srv/scratch/z5173707/phonological/librispeech/cache/'
libri = load_dataset('librispeech_asr',version, cache_dir=cache_dir)
#libri = DatasetDict({'train.100':libri[0],'validation':libri[1],'test':libri[2]})
phonemizer_dir = '/srv/scratch/z5173707/phonological/librispeech/phonemizer/'
phonemizer = Phonemizer.from_checkpoint(join(phonemizer_dir,'en_us_cmudict_forward.pt'))
cmu_dict = cmudict.dict()
def phonemize(batch):
    text = batch['text'].lower()
    phoneme_str=''
    for word in text.split():
        if word in cmu_dict:
            phoneme_str += re.sub('[0-9]','',' '.join(cmu_dict.get(word)[0]))
        else:
            phoneme_str += phonemizer(word, lang='en_us',expand_acronyms=False).replace('][',' ').replace(']','').replace('[','')
        phoneme_str += ' '
    phoneme_str = phoneme_str.lower()
    batch['phoneme'] = phoneme_str.strip()
    #batch['file_path'] = batch['file']
    return batch
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower() + " "
    return batch
def get_duration(batch):
    batch['duration'] = librosa.get_duration(filename=batch['file'])
    return batch
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = sf.read(batch["file"])
    batch["speech"] = speech_array
    batch["sampling_rate"] = sampling_rate
    return batch
nproc=30
print('remove_special_characters')
libri = libri.map(remove_special_characters, num_proc=nproc)
print('phonemizer')
libri = libri.map(phonemize)
print('get_duration')
libri = libri.map(get_duration,num_proc=nproc)
print('filtering')
libri = libri.filter(lambda x:x['duration'] < 15, num_proc=nproc)
libri = libri.map(speech_file_to_array_fn, num_proc=nproc)
libri.save_to_disk(save_dir)

