from torch.nn.functional import nll_loss, log_softmax
import torch

from fastai.text import RNNLearner
def annotation_loss(input, target):
    # need to make this work with nll loss
    # shape will now be bs x n_out(BIO) x num_anns x tl
    input = log_softmax(input,dim=-1)
    input = torch.transpose(input, 1, 3)
    target = torch.transpose(target, 1, 2)
    return nll_loss(input, target, ignore_index=-1, weight=torch.FloatTensor([1, 20]).cuda())