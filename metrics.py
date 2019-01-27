from fastai.text import fbeta

def recall(index, label):

    def recall_inner(input, targs):
        # input is bs x tl x 18 x 2
        input = input.argmax(-1)[:,:,index]
        input = (input > 0).long()
        # now it's bs x tl
        targs = targs[:,:,index]
        targs = (targs > 0).long()
        TP = (input * targs).sum(dim=1).float()
        rec = TP / (targs.sum(dim=1).float() + 1e-9)
        return rec.mean()
    recall_inner.__name__ = f'recall_{label}'
    return recall_inner


def precision(index, label):

    def precision_inner(input, targs):
        # input is bs x tl x 18 x 2
        input = input.argmax(-1)[:,:,index]
        input = (input > 0).long()
        # now it's bs x tl
        targs = targs[:,:,index]
        targs = (targs > 0).long()
        TP = (input * targs).sum(dim=1).float()
        prec = TP / (input.sum(dim=1).float() + 1e-9)
        return prec.mean()
    precision_inner.__name__ = f'precision_{label}'
    return precision_inner