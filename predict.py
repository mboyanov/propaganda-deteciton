import spacy
import torch
import pandas as pd

nlp = spacy.load('en')

def predict_task_2(ds, model, vocab):
    task2res = []
    for doc_id, doc in zip(ds.ids, ds.docs):
        for sentence_id, sentence in enumerate(doc.text.split("\n")):
            if sentence_id == 0:
                continue
            if sentence.strip() == "" or sentence_id == 0:
                task2res.append({'id': doc_id, 'sentence_id': sentence_id, 'propaganda': 'non-propaganda'})
                continue
            sentence_doc = nlp(sentence)
            sentence_tokens = [x.lower_ for x in sentence_doc]
            sentence_token_ids = vocab.numericalize(sentence_tokens)
            res = model(torch.LongTensor(sentence_token_ids).unsqueeze(0).cuda())
            res = res[0].argmax(-1).sum() > 0
            label = 'propaganda' if res > 0 else 'non-propaganda'
            task2res.append({'id': doc_id, 'sentence_id': sentence_id, 'propaganda': label})
    return pd.DataFrame(task2res)
