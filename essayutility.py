import json
import nltk
import re
import pprint
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.corpus import gutenberg
from urllib import request
from nltk.probability import FreqDist
from nltk.probability import ConditionalFreqDist
import random

with open('TeacherAI/tai-documents-v3.json') as f:
    data = json.load(f)

def lead_score(j):
    return j["grades"][1]["score"]["criteria"]["lead"]

def score_list(score):
    return [i for i in data if lead_score(i) == score]

# lead_counts = [str(lead_score(i)) for i in data]
# lead_counts = FreqDist(lead_counts)
# print(lead_counts.most_common(10))

def vocab_size(essay):
    words = re.split(r'\W+',essay["plaintext"])
    words = [w.lower() for w in words]
    return len(set(words))

# good_vocab_avg = [vocab_size(plaintext_prep(i)) for i in good_leads]
# good_vocab_avg = sum(good_vocab_avg) / len(good_vocab_avg)
# print(good_vocab_avg)
#
# bad_vocab_avg = [vocab_size(plaintext_prep(i)) for i in bad_leads]
# bad_vocab_avg = sum(bad_vocab_avg) / len(bad_vocab_avg)
# print(bad_vocab_avg)

def sentence_statistics(essay):
    """Explaratory Data Analysis Function"""
    temp = sent_tokenize(essay["plaintext"])
    temp = [word_tokenize(i) for i in temp]

    def average_sentence_length():
        return sum([len(i) for i in temp]) / len(temp)

    m = average_sentence_length()

    def sentence_length_variation():
        return sum([(len(i) - m)*(len(i) - m) for i in temp]) / len(temp)

    return (m,vocab_size(essay),)

def essay_features(essay):
    """Generate Feature Dictionary for Classifier"""
    stats = sentence_statistics(essay)
    return {"avg_sentence_length":stats[0],"vocab_size":stats[2]}

def print_stats(stats):
    print(sum([i[0] for i in stats]) / len(stats), " ", sum([i[1] for i in stats]) / len(stats),
          sum([i[2] for i in stats]) / len(stats))

# for i in [0.5 * i for i in range(2,9)]:
#     stats = [sentence_statistics(i) for i in score_list(i)]
#     print(str(i)," ",len(stats),": ")
#     print_stats(stats)

random.shuffle(data)
train = data[:170]
test = data[170:]

test = [(essay_features(e), lead_score(e)) for e in test]
train = [(essay_features(e), lead_score(e)) for e in train]

classifier = nltk.DecisionTreeClassifier.train(train,entropy_cutoff=0.1)
print("Accuracy: ", nltk.classify.accuracy(classifier, test))
