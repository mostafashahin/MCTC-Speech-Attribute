#Eval
import numpy as np
import pandas as pd
import jiwer.transforms as tr
from typing import Union, List, Tuple, Dict
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
from datasets import load_metric, load_dataset, load_from_disk
import re
import soundfile as sf
import torch
import Levenshtein
from collections import defaultdict
import sys

mode_dir = sys.argv[1]
result_dir = sys.argv[2]

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
#Mapper is a csv file with label,token

def compute_mertics(references: list, predictions: list, tokenizer, reference_phonemes: list = None, token_mapper=None) -> dict:
    offset = 0
    if tokenizer.unk_token_id:
        offset = 2
    else:
        offset = 1
    vocab_size = tokenizer.vocab_size -offset #exclude pad and unk
    conf_matrix = np.zeros((vocab_size, vocab_size))
    insertions = np.zeros(vocab_size)
    deletions = np.zeros(vocab_size)
    #compute SER
    ref_ar = np.asarray(references)
    pred_ar = np.asarray(predictions)
    total_cor = np.sum((ref_ar == pred_ar).astype(int))
    ser = 1.0-(total_cor/len(ref_ar))
    if reference_phonemes:
        per_phoneme = defaultdict(lambda: defaultdict(int))
        phoneme2att = {}

    att_labels = tokenizer.convert_ids_to_tokens(np.arange(vocab_size)+offset)
    if token_mapper:
        with open(token_mapper,'r') as f:
            mapper = dict([line.split(',')[::-1] for line in f.read().splitlines()])
        att_labels = [mapper[l] for l in att_labels]

    #for truth, hypothesis in zip(references,predictions):
    for indx in range(len(references)):
        truth, hypothesis = references[indx], predictions[indx]
        if reference_phonemes:
            truth_phoneme = reference_phonemes[indx]
        #print(truth, len(truth.split()))
        #print(hypothesis,len(hypothesis.split()))

        truth, hypothesis = preprocess(truth, hypothesis, default_transform, default_transform, tokenizer)
        if reference_phonemes:
            assert len(truth) == len(truth_phoneme)
        for i in range(len(truth)):
        
            truth_char_int = ord(truth[i])
            conf_matrix[truth_char_int,truth_char_int] +=1
            if reference_phonemes:
                truth_att = tokenizer.convert_ids_to_tokens(truth_char_int+offset)
                if token_mapper:
                    truth_att = mapper[truth_att]
                phoneme2att[truth_phoneme[i]] = truth_att
            
        if reference_phonemes:
            for p in truth_phoneme:
                per_phoneme[p]['hit'] += 1
        ops = Levenshtein.editops(truth, hypothesis)

    #print(truth, len(truth))
    #print(hypothesis,len(hypothesis))

        for op in ops:
            #print(op)
            if op[0] == 'replace':
                truth_char_int = ord(truth[op[1]])
                hypothesis_char_int = ord(hypothesis[op[2]])
                conf_matrix[hypothesis_char_int,truth_char_int] += 1
                conf_matrix[truth_char_int,truth_char_int] -= 1
                if reference_phonemes:
                    p = truth_phoneme[op[1]]
                    per_phoneme[p]['hit'] -= 1
                    pred_as = tokenizer.convert_ids_to_tokens(hypothesis_char_int+offset)#predictions[indx][op[2]]
                    if token_mapper:
                        pred_as = mapper[pred_as]
                    per_phoneme[p][pred_as] += 1
            
            elif op[0] == 'insert':
                hypothesis_char_int = ord(hypothesis[op[2]])
                insertions[hypothesis_char_int] += 1
            elif op[0] == 'delete':
                truth_char_int = ord(truth[op[1]])
                deletions[truth_char_int] += 1
                conf_matrix[truth_char_int,truth_char_int] -= 1
                if reference_phonemes:
                    p = truth_phoneme[op[1]]
                    per_phoneme[p]['hit'] -= 1
                    per_phoneme[p]['del'] += 1
    #Recall is the TP/TP+FN deletion is included in the FN 
    recall = conf_matrix.diagonal() / (conf_matrix.sum(axis=0) + deletions)
    #Precition is the TP/TP+FP insertion is included in the FP
    precision = conf_matrix.diagonal() / (conf_matrix.sum(axis=1) + insertions)

    metrics = {'att_labels':att_labels,'conf_matrix':conf_matrix, 'insertions':insertions, 'deletions':deletions, 'recall':recall,'precision':precision, 'ser':ser}
    if reference_phonemes:
        metrics['per_phoneme'] = per_phoneme
        metrics['phoneme2att'] = phoneme2att
    return metrics


