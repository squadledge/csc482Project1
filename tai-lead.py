import json
import nltk
import random
import re
import sys


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


# returns the lead paragraph (list of list of tokens)
def get_lead(raw_text):
    for i in range(4, 0, -1):
        for paragraph in raw_text:
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


# returns most common pos of first word in sentences
def get_sentence_starter_percents(paragraph):
    starter_list = []
    for sentence in paragraph:
        starter_list.append(nltk.pos_tag(sentence)[0][1])
    starter_counts = dict()
    for i in starter_list:
        starter_counts[i] = starter_counts.get(i, 0) + 1
    return starter_counts


def get_features_dict(text):
    initial = {
        'num_sentences': len(text),
        'avg_num_words_per_sentence': get_average_sentence_length(text),
        'vocab_size': get_vocab_size(text),
        'sentence_length_range': get_sentence_length_range(text),
        'longest_sentence_length': get_length_of_longest_sentence(text),
    }
    initial.update(get_sentence_starter_percents(text))
    return initial


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
            lambda essay_tuple: (get_features_dict(get_lead(essay_tuple[1])), get_score(essay_tuple[0], LEAD)),
            essay_list
        )
    )

    random.shuffle(feature_score_list)
    # 210 total in feature_score_list
    train_set = feature_score_list[:170]
    test_set = feature_score_list[170:]

    # cross-validation
    num_folds = 10
    subset_size = int(len(train_set) / num_folds)
    accuracies = []

    for i in range(num_folds):
        print("Round ", i)
        testing_this_round = train_set[i * subset_size:][:subset_size]
        training_this_round = train_set[:i * subset_size] + train_set[(i + 1) * subset_size:]
        # train using training_this_round
        # evaluate against testing_this_round
        # save accuracy
        classifier = nltk.DecisionTreeClassifier.train(training_this_round)
        accuracies.append(nltk.classify.accuracy(classifier, testing_this_round))

    print('K-Fold Cross Validation Accuracy: {}'.format(sum(accuracies) / len(accuracies)))
    print('Accuracy: {}'.format(nltk.classify.accuracy(classifier, test_set)))

    classifier = nltk.DecisionTreeClassifier.train(feature_score_list)

    with open(sys.argv[1]) as input_file:
        input_tokens = tokenized_text(input_file.read())
        print('Predicted Lead Score: {}'.format(
            classifier.classify(get_features_dict(get_lead(input_tokens)))))
