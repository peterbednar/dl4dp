from dl4dp import main

import torch
from .modules import LSTM, unbind_sequence
if __name__ == "__main__":
    #main()
    batch = [torch.rand((10,4)), torch.rand((8,4)), torch.rand((5,4)), torch.rand((11,4))]
    lstm = LSTM(4, 6, 1)
    h, _, lengths = lstm(batch, unpad=True)

    h = torch.cat(h)
    h = unbind_sequence(h, lengths)
    print([x.size() for x in h])