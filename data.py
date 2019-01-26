from collections import defaultdict
import pathlib
import spacy
from fastai.text import *
from spacy.tokens import Doc, Token
from keras.utils.np_utils import to_categorical

nlp = spacy.load('en')


def parse_label(label_path):
    # idx, type, start, end
    labels = []
    for line in open(label_path):
        parts = line.strip().split('\t')
        labels.append((int(parts[2]), int(parts[3]), parts[1]))
    return sorted(labels)


def read_data(directory):
    texts = []
    labels = []

    for f in directory.glob('*.txt'):
        texts.append(f.read_text())
        labels.append(parse_label(f.as_posix().replace('.txt', '.task3.labels')))
    docs = list(nlp.pipe(texts))

    return docs, labels


PROPAGANDA_TYPES = ["Appeal_to_Authority",
                    "Appeal_to_fear-prejudice",
                    "Bandwagon",
                    "Black-and-White_Fallacy",
                    "Causal_Oversimplification",
                    "Doubt",
                    "Exaggeration,Minimisation",
                    "Flag-Waving",
                    "Loaded_Language",
                    "Name_Calling,Labeling",
                    "Obfuscation,Intentional_Vagueness,Confusion",
                    "Red_Herring",
                    "Reductio_ad_hitlerum",
                    "Repetition",
                    "Slogans",
                    "Straw_Men",
                    "Thought-terminating_Cliches",
                    "Whataboutism"
                    ]

PT2ID = {y: x for (x, y) in enumerate(PROPAGANDA_TYPES)}

BEGIN = 1
INSIDE = 2


def _numericalize_labels_single(doc: Doc, doc_labels: list):
    res = np.zeros((len(doc), len(PROPAGANDA_TYPES)))
    token_idx = 0
    labels_idx = 0
    while labels_idx < len(doc_labels):
        current_token: Token = doc[token_idx]
        current_label = doc_labels[labels_idx]
        # advance token until it is within the label
        if current_token.idx < current_label[0]:
            token_idx += 1
            continue
        # mark all tokens in the span as B or I
        start_token_idx = token_idx

        while current_token.idx < current_label[1]:
            label = BEGIN if current_token.i == start_token_idx else INSIDE
            res[token_idx, PT2ID[current_label[2]]] = label
            token_idx += 1
            if token_idx >= len(doc):
                break
            current_token = doc[token_idx]
        # advance label
        labels_idx += 1
        # revert token_idx because the labels might be intersecting
        token_idx = start_token_idx

    return to_categorical(res, 3)


def numericalize_labels(docs, labels):
    """
    For each document creates a numpy array of shape (tl, 18, 3) containing BIO annotations for the tokens
    :param docs: Docs are a list of spacy documents.
    :param labels: Labels are a list of sorted triples (start, end, TYPE)
    :return:
    """
    return [_numericalize_labels_single(d, l) for d, l in zip(docs, labels)]


class PropagandaDataset(Dataset):

    def __init__(self, docs: list, labels: list, vocab: Vocab):
        super().__init__()
        self.docs = docs
        self.labels = labels
        self.tokens = [[t.lower_ for t in doc] for doc in docs]
        # TODO: get token spans
        self.token_spans = None
        self.labels_npy = numericalize_labels(docs, labels)
        self.numericalized_tokens = [vocab.numericalize(row) for row in self.tokens]

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, item):
        return self.numericalized_tokens[item], self.labels_npy[item]


if __name__ == '__main__':
    directory = pathlib.Path('/data/fakenews/tasks-2-3/train/')
    vocab = None
    texts, labels = read_data(directory)
    train = PropagandaDataset(texts, labels)

    db = DataBunch(DataLoader(train, batch_size=8), DataLoader(train))

    vocab_sz = 60000
    emb_sz = 300
    n_hid = 300
    n_layers = 2
    pad_token = 1
    model = SequentialRNN(MultiBatchRNNCore(120, 10000, vocab_sz, emb_sz, n_hid, n_layers, pad_token),
                          LinearDecoder(3, n_hid, 0.15))

    learner = RNNLearner(db, model)
    learner.fit_one_cycle(1, 3e-3)
