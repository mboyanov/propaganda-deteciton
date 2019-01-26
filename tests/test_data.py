from fastai.text import Vocab
from data import PropagandaDataset, PROPAGANDA_TYPES
import spacy
import numpy as np
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


from keras.utils.np_utils import to_categorical

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
    expected = to_categorical(expected, 3)
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
    expected = to_categorical(expected, 3)
    np.testing.assert_array_equal(ys, expected)