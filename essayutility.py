import json
import nltk
import random
import re
import sys

from sklearn.feature_extraction import DictVectorizer


LEAD = 0
ENDING = 1
CHOICE = 2


# returns tokenized formatted as: Paragraph List -> Sentence List -> Word List
def tokenized_text(raw_text):
    return [[nltk.word_tokenize(s)
             for s in sent_tokenizer.tokenize(par.strip())]
            for par in raw_text.split("\n\n")]


# returns most recent grade dict from grade list
def get_recent_version(grade_list):
    return max(grade_list, key=lambda grade: grade['version'])


def get_score(score_object, criteria):
    if criteria == LEAD:
        return score_object['score']['criteria']['lead']
    elif criteria == ENDING:
        return score_object['score']['criteria']['ending']
    else:
        return score_object['score']['criteria']['spelling']


def to_single_paragraph(text):
    return [sentence for paragraph in text for sentence in paragraph]


def paragraph_to_words(paragraph):
    return [word.lower() for sentence in paragraph for word in sentence]


def get_vocab_size(paragraph):
    words = paragraph_to_words(paragraph)
    vocab = filter(lambda word: re.match(r"^[a-z]", word), words)
    return len(set(vocab))


def get_average_word_length(paragraph):
    words = paragraph_to_words(paragraph)
    return sum(map(len, words)) / len(words)


def get_percent_vowels(paragraph):
    words = paragraph_to_words(paragraph)
    letters = [letter for word in words for letter in word]
    vowel_list = ['a', 'e', 'i', 'o', 'u']
    return (len(list(filter(lambda letter: letter.lower() in vowel_list, letters)))
            / len(letters))


def get_features_dict(text):
    return {
        'vocab_size': get_vocab_size(text),
        'avg_word_length': get_average_word_length(text),
        'percent_vowels': get_percent_vowels(text)
    }

sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

with open('./TeacherAI/tai-documents-v3.json') as essay_json_file:
    # list of essays and grades
    # [{id: ID, plaintext: PT, doctitle: DT, grades: G}, ...]
    json_list = json.load(essay_json_file)

    # grades: [{version: V, comment: c, score: s, markup: M, checkboxes: C}, ...]
    # score: {total: T, average: A, criteria: {overall: O, lead: L, ending: E, punctuation: P, ...}, ...}
    # essay_list is list of tuples: (most recent grade dict, tokenized text)
    essay_list = list(map(lambda d: (get_recent_version(d['grades']), tokenized_text(d['plaintext'])), json_list))

    feature_score_list = list(
        map(
            lambda essay_tuple: (get_features_dict(to_single_paragraph(essay_tuple[1])), get_score(essay_tuple[0], CHOICE)),
            essay_list
        )
    )

    random.shuffle(feature_score_list)

    X_train = [i[0] for i in feature_score_list]
    X_train = [[i["vocab_size"],i["avg_word_length"],i["percent_vowels"]] for i in X_train]
    Y_train = [str(i[1]) for i in feature_score_list]

    X_test = X_train[170:]
    Y_test = Y_train[170:]

    X_train = X_train[:170]
    Y_train = Y_train[:170]

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    import numpy as np
    from sklearn.metrics import confusion_matrix, classification_report, precision_score, accuracy_score

    lda = LinearDiscriminantAnalysis()
    model = lda.fit(X_train,Y_train)

    pred = model.predict(X_test)
    pred = model.predict(X_test)
    
    print(accuracy_score(Y_test, pred))





