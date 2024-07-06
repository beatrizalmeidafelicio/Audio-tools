import numpy as np
import re
import torch
from num2words import num2words
from torchmetrics.text import CharErrorRate
import matplotlib.pyplot as plt

alphabet = 'abcdefghijklmnopqrstuvwxyzáéíóúâêîôûãõàèìòùç '
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# normalizar o texto
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

def read_file_as_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().replace('\n', ' ')


pred_file_path = 'C:\\Users\\BIA\\Desktop\\ground truth\\fatima-roda-viva\\parte4-wav2vec.txt'
target_file_path = 'C:\\Users\\BIA\\Desktop\\novas transcricoes\\podcast_parte4.txt'

preds = normalize(read_file_as_text(pred_file_path))
targets = normalize(read_file_as_text(target_file_path))


cer = CharErrorRate()
cer.update([preds], [targets])
cer_value = cer.compute()
print(f"Character Error Rate (CER): {cer_value.item()}")
