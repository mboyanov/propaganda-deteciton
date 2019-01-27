import spacy.lang.en
from spacy.tokens import Doc, Token
from collections import Counter
import numpy as np
from utils import get_consecutive_spans, get_char_spans

stop_words = spacy.lang.en.stop_words.STOP_WORDS.union(
    {'.', '\n', 'the', '?', ',', '\n\n', '\n', '-', '"', '\'s', '’', '_', '%', '(', ')',
     '”', '“', '[', ']'})


def not_all_stops(ngram):
    return np.any([x not in stop_words for x in ngram.split(';')])


def get_grams(doc):
    unigram_counts = Counter()
    bigram_counts = Counter()
    trigram_counts = Counter()
    for t1, t2 in zip(doc, doc[1:]):
        unigram_counts[t1.lemma_] += 1
        bigram_counts[_key(t1, t2)] += 1
    for t1, t2, t3 in zip(doc, doc[1:], doc[2:]):
        trigram_counts[_key(t1, t2, t3)] += 1
    return [x for x, y in bigram_counts.most_common(100) if not_all_stops(x) and bigram_counts[x] >= 3][:5], \
           [x for x, y in unigram_counts.most_common(100) if x not in stop_words and y > 4][:5], \
           [x for x, y in trigram_counts.most_common(100) if not_all_stops(x) and trigram_counts[x] >= 4][:5]



def detect_repetition(doc):
    unigrams, bigrams, trigrams = get_grams(doc)
    repetition_marks = mark(doc, unigrams, bigrams, trigrams)
    spans = get_consecutive_spans(repetition_marks)
    char_spans = get_char_spans(spans, doc)
    return char_spans


def _key(*tuple):
    return ";".join(x.lower_ for x in tuple)


def mark(doc, unigrams, bigrams, trigrams):
    res = [0] * len(doc)
    for i, (t1, t2, t3) in enumerate(zip(doc, doc[1:], doc[2:])):
        if t1.lower_ in unigrams:
            res[i] = 1
        if _key(t1, t2) in bigrams:
            res[i: i + 1] = [1, 1]
        if _key(t1, t2, t3) in trigrams:
            res[i:i + 2] = [1, 1, 1]
    return res

# nlp = spacy.load('en')
# doc = nlp("law-abiding citizens is the law-abiding citizens law-abiding citizens")
# print(repetition_detector(doc))
