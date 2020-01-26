import json
import nltk
import re
import pprint
from nltk import word_tokenize
from nltk.corpus import gutenberg
from urllib import request
from nltk.probability import FreqDist
from nltk.probability import ConditionalFreqDist
import random

with open('TeacherAI/tai-documents-v3.json') as f:
    data = json.load(f)

good_leads = [i for i in data if i["grades"][1]["score"]["criteria"]["lead"] >= 4.0]
bad_leads = [i for i in data if i["grades"][1]["score"]["criteria"]["lead"] == 1.0]

def lead_score(j):
    return j["grades"][1]["score"]["criteria"]["lead"]

# lead_counts = [str(lead_score(i)) for i in data]
# lead_counts = FreqDist(lead_counts)
# print(lead_counts.most_common(8))
def plaintext_prep(essay):
    return word_tokenize(essay["plaintext"])

def vocab_size(words):
    return len(set(words))

good_vocab_avg = [vocab_size(plaintext_prep(i)) for i in good_leads]
good_vocab_avg = sum(good_vocab_avg) / len(good_vocab_avg)
print(good_vocab_avg)

bad_vocab_avg = [vocab_size(plaintext_prep(i)) for i in bad_leads]
bad_vocab_avg = sum(bad_vocab_avg) / len(bad_vocab_avg)
print(bad_vocab_avg)