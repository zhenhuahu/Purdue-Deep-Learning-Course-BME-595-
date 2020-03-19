import torch
from torch.distributions import normal


class NeuralNetwork(object):

    def __init__(self, list_in):  # list_in is the input list.[in, h1, ..., hn, out]
        self.list_in = list_in
        self.szLayer = len(list_in)  # length of input list
        # initialize theta matrix based on layer size
        self.theta = [(normal.Normal(torch.tensor(0.0), torch.rsqrt((torch.tensor(x).float())))).sample((x + 1, y)) for x, y in zip(list_in[:-1], list_in[1:])]

    def getLayer(self, layer):  # layer index starts from 0
        return self.theta[layer]

    def forward(self, input):
        bias = torch.ones([1, input.size()[1]])
        # biased input
        bias_input = torch.cat((bias, input), 0)  # 3*n
        bias_input = bias_input.t()  # transpose

        # do forward calculation
        for i in range(0, self.szLayer - 1):
            tmpVec = torch.mm(bias_input, self.theta[i])
            # sigmoid function
            tmpVec = 1.0 / (1.0 + torch.exp(0 - tmpVec))
            # print(tmpVec)
            bias_input = torch.cat((bias.t(), tmpVec), 1)
            # print(bias_input)

        out = tmpVec
        out = out.t()

        return out
