import torch
from fastai.text import Vocab
from data import PropagandaDataset, PROPAGANDA_TYPES
import spacy
import numpy as np
from data import pad_collate_concat
nlp = spacy.blank('en')


def test_should_annotate_tokens():
    # GIVEN
    vocab = Vocab(['_unk_','_pad_', 'hello', 'world'])
    text = "hello world"
    docs = list(nlp.pipe([text]))
    # WHEN
    ds = PropagandaDataset(docs, [{}], vocab)
    # THEN
    np.testing.assert_array_equal(ds[0][0], [2, 3])



def test_should_annotate_labels():
    # GIVEN
    vocab = Vocab(['_unk_','_pad_', 'hello', 'world'])
    text = "hello world"
    docs = list(nlp.pipe([text]))
    # WHEN
    ds = PropagandaDataset(docs, [
        [(0,5, PROPAGANDA_TYPES[0])]
    ], vocab)
    # THEN
    item = ds[0]
    ys = item[1]

    expected = np.zeros((len(docs[0]), len(PROPAGANDA_TYPES)))
    expected[0, 0] = 1
    np.testing.assert_array_equal(ys, expected)



def test_should_annotate_labels__inside():
    # GIVEN
    vocab = Vocab(['_unk_','_pad_', 'hello', 'world'])
    text = "hello world"
    docs = list(nlp.pipe([text]))
    # WHEN
    ds = PropagandaDataset(docs, [
        [(0, 11, PROPAGANDA_TYPES[0])]
    ], vocab)
    # THEN
    item = ds[0]
    ys = item[1]

    expected = np.zeros((len(docs[0]), len(PROPAGANDA_TYPES)))
    expected[0, 0] = 1
    expected[1, 0] = 2
    np.testing.assert_array_equal(ys, expected)



def test_pad_collate_xs_ys():
    # GIVEN
    samples = [
        (np.array([2,3,4]), np.zeros((3, len(PROPAGANDA_TYPES)))),
        (np.array([2, 3]), np.zeros((2, len(PROPAGANDA_TYPES))))
    ]
    samples[0][1][1, 2] = 1

    # WHEN
    res = pad_collate_concat(samples)
    # THEN
    assert res[0].size() == torch.Size((2, 3))
    assert res[1].size() == torch.Size((2, 3, len(PROPAGANDA_TYPES)))
    assert res[1][0, 1,2] == 1