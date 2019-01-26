from data import PROPAGANDA_TYPES
from output import get_consecutive_spans, output_single
import spacy
import numpy as np
nlp = spacy.blank('en')

def test_consecutive_spans_simple():
    # GIVEN
    arr = [0,1,1,0]
    # WHEN
    res = get_consecutive_spans(arr)
    # THEN
    assert res[0] == (1, 3)


def test_consecutive_spans_end():
    # GIVEN
    arr = [0,1,1,1]
    # WHEN
    res = get_consecutive_spans(arr)
    # THEN
    assert res[0] == (1, 4)


def test_output_single():
    # GIVEN
    doc = nlp("I like the way you move")
    scores = np.zeros((6, 18))
    scores[1, 8] = 1
    scores[2, 8] = 1
    # WHEN
    res = output_single('123', doc, scores)
    # THEN
    assert res is not None
    assert res.iloc[0]['id'] == '123'
    assert res.iloc[0]['start'] == 2
    assert res.iloc[0]['end'] == 10
    assert res.iloc[0]['propaganda'] == PROPAGANDA_TYPES[8]
