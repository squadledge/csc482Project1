import json
import nltk
import random
import re
import sys
from nltk.corpus import cmudict


LEAD = 0
ENDING = 1
CHOICE = 2
d = cmudict.dict()

# returns tokenized formatted as: Paragraph List -> Sentence List -> Word List
def tokenized_text(raw_text):
    return [[nltk.word_tokenize(s)
             for s in sent_tokenizer.tokenize(par.strip())]
            for par in raw_text.split("\n\n")]


# returns most recent grade dict from grade list
def get_recent_version(grade_list):
    return max(grade_list, key=lambda grade: grade['version'])


# returns the ending paragraph (list of list of tokens)
def get_ending(raw_text):
    for i in range(4, 0, -1):
        for paragraph in raw_text[::-1]:
            if len(paragraph) >= i:
                return paragraph
    raise Exception('No paragraphs with sentences')


def get_score(score_object, criteria):
    if criteria == LEAD:
        return score_object['score']['criteria']['lead']
    elif criteria == ENDING:
        return score_object['score']['criteria']['ending']
    else:
        return score_object['score']['criteria']['spelling']


def paragraph_to_words(paragraph):
    return [word.lower() for sentence in paragraph for word in sentence]


def get_average_sentence_length(paragraph):
    return sum(map(lambda sentence: len(sentence), paragraph)) / len(paragraph)


def get_vocab_size(paragraph):
    words = paragraph_to_words(paragraph)
    vocab = filter(lambda word: re.match(r"^[a-z]", word), words)
    return len(set(vocab))


def get_average_word_length(paragraph):
    words = paragraph_to_words(paragraph)
    return sum(map(len, words)) / len(words)


def get_length_of_longest_sentence(paragraph):
    sentence_lengths = list(map(len, paragraph))
    return max(sentence_lengths)


def get_length_of_shortest_sentence(paragraph):
    sentence_lengths = list(map(len, paragraph))
    return min(sentence_lengths)


def get_sentence_length_range(paragraph):
    max_length = get_length_of_longest_sentence(paragraph)
    min_length = get_length_of_shortest_sentence(paragraph)
    return max_length - min_length

def syllable_word_ratio(paragraph):
    def nsyl(word):
        try:
            return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
        except:
            return 0
    return sum([nsyl(word) for sen in paragraph for word in sen]) / sum([len(sen) for sen in paragraph])

def get_features_dict(text):
    return {
        'num_sentences': len(text),
        'avg_num_words_per_sentence': get_average_sentence_length(text),
        'vocab_size': get_vocab_size(text),
        'avg_word_length': get_average_word_length(text),
        'sentence_length_range': get_sentence_length_range(text),
        'longest_sentence_length': get_length_of_longest_sentence(text),
        'syllable_word_ratio': syllable_word_ratio(text)
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
            lambda essay_tuple: (get_features_dict(get_ending(essay_tuple[1])), get_score(essay_tuple[0], ENDING)),
            essay_list
        )
    )

    random.shuffle(feature_score_list)

    X_train = [i[0] for i in feature_score_list]
    X_train = [[i["num_sentences"],
                i["avg_num_words_per_sentence"],
                i["vocab_size"],
                i['avg_word_length'],
                i['sentence_length_range'],
                i['longest_sentence_length'],
                i['syllable_word_ratio']] for i in X_train]
    Y_train = [str(i[1]) for i in feature_score_list]

    X_test = X_train[170:]
    Y_test = Y_train[170:]

    X_train = X_train[:170]
    Y_train = Y_train[:170]

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    import numpy as np
    from sklearn.metrics import confusion_matrix, classification_report, precision_score, accuracy_score

    lda = LinearDiscriminantAnalysis()
    model = lda.fit(X_train, Y_train)

    pred = model.predict(X_test)

    # print(accuracy_score(Y_test, pred))

    with open(sys.argv[1]) as input_file:
        input_tokens = tokenized_text(input_file.read())
        i = get_features_dict(get_ending(input_tokens))
        features = [[i["num_sentences"],
                i["avg_num_words_per_sentence"],
                i["vocab_size"],
                i['avg_word_length'],
                i['sentence_length_range'],
                i['longest_sentence_length'],
                i['syllable_word_ratio']]]
        print('Predicted Ending Score: {}'.format(
            model.predict(features)[0]))
