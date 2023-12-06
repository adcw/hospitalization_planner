import torch
from torch import nn
from torch.autograd import Variable


class GaussianDropout(nn.Module):
    def __init__(self, alpha=1.0):
        super(GaussianDropout, self).__init__()
        self.alpha = torch.cuda.FloatTensor([alpha])

    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            epsilon = torch.cuda.FloatTensor(*x.size()).normal_() * self.alpha + 1

            epsilon = Variable(epsilon)

            return x * epsilon
        else:
            return x
