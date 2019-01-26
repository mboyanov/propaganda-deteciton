

def recall(index):

    def recall_inner(input, targs):
        # input is bs x tl x 18 x 2
        input = input.argmax(-1)[:,:,index]
        # now it's bs x tl
        targs = targs[:,:,index]
        targs = (targs > 0).long()
        TP = (input * targs).sum(dim=1).float()
        rec = TP / (targs.sum(dim=1).float() + 1e-9)
        return rec.mean()
    recall_inner.__name__ = f'recall_{index}'
    return recall_inner