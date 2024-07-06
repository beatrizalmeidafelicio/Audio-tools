import numpy as np
import re
import torch
from num2words import num2words


alphabet = 'abcdefghijklmnopqrstuvwxyzáéíóúâêîôûãõàèìòùç '
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize(phrase):
    phrase = phrase.lower()
   
    new_words = []
    for word in phrase.split():
        word = re.sub(r"\d+[%]", lambda x: x.group() + " por cento", word)
        word = re.sub(r"%", "", word)
        word = re.sub(r"\d+[o]{1}", lambda x: num2words(x.group()[:-1], to='ordinal', lang='pt_BR'), word)
        ref = word
        word = re.sub(r"\d+[a]{1}", lambda x: num2words(x.group()[:-1], to='ordinal', lang='pt_BR'), word)
        if word != ref:
            segs = word.split(' ')
            word = ''
            for seg in segs:
                word += seg[:-1] + 'a' + ' '
            word = word[:-1]


        if any(i.isdigit() for i in word):
            segs = re.split(r"[?.!\s]", word)
            word = ''
            for seg in segs:
                if seg.isnumeric():
                    seg = num2words(seg, lang='pt_BR')
                word += seg + ' '
            word = word[:-1]
        new_words.append(word)
   
    phrase = ' '.join(new_words)
   
    for c in phrase:
        if c not in alphabet:
            phrase = phrase.replace(c, '')


    return phrase


def calculate_wer(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    for j in range(len(hyp_words) + 1):
        d[0, j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                substitution = d[i - 1, j - 1] + 1
                insertion = d[i, j - 1] + 1
                deletion = d[i - 1, j] + 1
                d[i, j] = min(substitution, insertion, deletion)
    wer = d[len(ref_words), len(hyp_words)] / len(ref_words)
    return wer


def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


reference_file_path = 'C:\\Users\\BIA\\Desktop\\ground truth\\fatima-roda-viva\\parte4-wav2vec.txt'
hypothesis_file_path = 'C:\\Users\\BIA\\Desktop\\novas transcricoes\\podcast_parte4.txt'


reference_text = load_text_file(reference_file_path)
hypothesis_text = load_text_file(hypothesis_file_path)


reference_text = normalize(reference_text)
hypothesis_text = normalize(hypothesis_text)


# calcular WER
wer_score = calculate_wer(reference_text, hypothesis_text)
print("Word Error Rate (WER):", wer_score)



