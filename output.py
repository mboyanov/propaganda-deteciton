from data import PropagandaDataset, PROPAGANDA_TYPES
import pandas as pd
import fastprogress
from repetiton_detector import detect_repetition
from utils import get_consecutive_spans, get_char_spans

HITLERUM = {'hitler','nazi', 'nazis', 'nazism',
                        'stalin', 'stalinism','stalinist', 'communist',
                        'fascist', 'fascism'}


def reduction(doc):
    reduction_spans = []
    for t in doc:
        if t.lower_ in HITLERUM:
            reduction_spans.append((t.idx, t.idx+len(t)))
    return reduction_spans


def output_single(id, doc, score):
    # score is tl x 18
    res = []
    for propaganda_type_id in range(len(PROPAGANDA_TYPES)):
        propaganda_type = PROPAGANDA_TYPES[propaganda_type_id]
        type_scores = score[:, propaganda_type_id]
        spans = get_consecutive_spans(type_scores)
        char_spans = get_char_spans(spans, doc)

        for cs in char_spans:
            res.append({
                'id': id,
                'propaganda': propaganda_type,
                'start': cs[0],
                'end': cs[1],
                'span': cs[2]
            })
    char_spans = reduction(doc)
    for cs in char_spans:
        res.append({
            'id': id,
            'propaganda': "Reductio_ad_hitlerum",
            'start': cs[0],
            'end': cs[1]
        })
    # repetition_spans = detect_repetition(doc)
    # for cs in repetition_spans:
    #     res.append({
    #         'id': id,
    #         'propaganda': "Repetition",
    #         'start': cs[0],
    #         'end': cs[1]
    #     })
    return pd.DataFrame(res)


def output_preds(ds: PropagandaDataset, scores):
    dfs = []
    for id, doc, score in fastprogress.progress_bar(zip(ds.ids, ds.docs, scores), total=len(ds)):
        dfs.append(output_single(id, doc, score))
    return pd.concat(dfs)
