from data import PropagandaDataset, PROPAGANDA_TYPES
import pandas as pd


def get_consecutive_spans(arr):
    span_len = 0
    spans = []
    for i, v in enumerate(arr):
        if v > 0:
            span_len += 1
        elif span_len > 0:
            spans.append((i - span_len, i))
            span_len = 0
    if span_len > 0:
        spans.append((len(arr) - span_len, len(arr)))
    return spans


def get_char_spans(spans, doc):
    def get_char_span(span):
        start_token = doc[span[0]]
        end_token = doc[span[1] - 1]
        if start_token.i == end_token.i:
            if start_token.lower_ in {'-', 'of', 'them','the','in', 'and', 'from', ',', 'they','is','to','who','that', 'must','.', 'its','a','an', 'not', 'do','“', "\"", 'or','it', 'on','i'}:
                return None
        # # expand around certain characters
        # if start_token.lower_ in {'-','\'s'}:
        #     start_idx = max(0, span[0] - 1)
        #     end_idx = min(len(doc)-1, span[1])
        #     start_token = doc[start_idx]
        #     end_token = doc[end_idx]
        #     print("expanded", doc[start_idx: end_idx+1])
        start_char = start_token.idx
        end_char = end_token.idx + len(end_token)
        return start_char, end_char, doc.char_span(start_char, end_char)

    char_spans = [get_char_span(s) for s in spans]
    return [cs for cs in char_spans if cs is not None]

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
    return pd.DataFrame(res)

import fastprogress
def output_preds(ds: PropagandaDataset, scores):
    dfs = []
    for id, doc, score in fastprogress.progress_bar(zip(ds.ids, ds.docs, scores), total=len(ds)):
        dfs.append(output_single(id, doc, score))
    return pd.concat(dfs)
