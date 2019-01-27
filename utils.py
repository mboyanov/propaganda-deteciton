
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
            if start_token.lower_ in {'-', 'of', 'them','the','in', 'my', 'no',
                                      'and', 'from', ',', 'they','is','to','who','that', 'must','.', 'its','a','an', 'not', 'do','“', "\"", 'or','it', 'on','i'}:
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