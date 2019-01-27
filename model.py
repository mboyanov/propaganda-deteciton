from torch import nn
from fastai.text.models import LinearDecoder
import torch
import torch.nn.functional

class MultiLinearDecoder(nn.Module):


    def __init__(self,num_anns, n_out, n_hid, output_p=0.15):
        super().__init__()
        self.decoders = nn.ModuleList([LinearDecoder(n_out, n_hid, output_p) for _ in range(num_anns)])


    def forward(self, input):
        raw_outputs, outputs = input
        all_activs = [decoder(input)[0].unsqueeze(-2) for decoder in self.decoders]
        # shape bs x tl x num_anns x n_out
        res = torch.cat(all_activs, -2)
        return res, raw_outputs, outputs