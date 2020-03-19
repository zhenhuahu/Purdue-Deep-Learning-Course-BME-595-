import torch
from torch.distributions import normal
from torch.autograd import Variable


class NeuralNetwork(object):

    def __init__(self, list_in):  # list_in is the input list.[in, h1, ..., hn, out]
        self.list_in = list_in
        self.szLayer = len(list_in)  # length of input list
        self.Theta = {}  # empty dictionary, weights between layers
        self.dE_dTheta = {}  # partial derivatives of Error on Thetas

        self.z = {}  # linear input to a node
        self.a = {}  # activation: output of nodes. a = sigma(z)
        #self.loss = 0.0

        # initialize theta matrix based on layer size
        self.keys = ["" for i in range(self.szLayer - 1)]
        for i in range(self.szLayer - 1):
                # keys
            self.keys[i] = "layer" + str(i) + str(i + 1)
            # values
            self.Theta[self.keys[i]] = (normal.Normal(torch.tensor(0.0), torch.rsqrt((torch.tensor(list_in[i]).float())))).sample((list_in[i + 1], list_in[i] + 1))

        # initialize theta matrix based on layer size
        # self.theta = [(normal.Normal(torch.tensor(0.0), torch.rsqrt((torch.tensor(x).float())))).sample((x + 1, y)) for x, y in zip(list_in[:-1], list_in[1:])]

    def getLayer(self, layer):  # layer index starts from 0
        return self.Theta[self.keys[layer]]

    def forward(self, inVal):
        self.a[0] = inVal
        bias = torch.ones([1, inVal.size()[1]])

        # do forward calculation
        for i in range(0, self.szLayer - 1):
            # biased input
            bias_input = torch.cat((bias, self.a[i]), 0)  # 3*n
            a_hat = bias_input  # transpose
            self.z[i + 1] = torch.mm(self.Theta["layer" + str(i) + str(i + 1)], a_hat)
            # sigmoid function
            self.a[i + 1] = torch.sigmoid(self.z[i + 1])
            # a_hat = torch.cat((bias.t(), tmpVec), 1)

        self.output = self.a[self.szLayer - 1]
        #out = out.t()
        return self.output

    def backward(self, target):
        # target: 1D FloatTensor. The supervised output value
        L = self.szLayer - 1  # output layer number
        m = target.size()[1]
        bias = torch.ones([1, m])
        # loss function
        self.loss = (((self.a[L] - target)**2).sum()*0.5)/m
        dE_da = self.a[L] - target
        dE_da = torch.sum(dE_da, dim = 0) / m
        #print('de_da size is {}'.format(dE_da.size()))

        dSigma_zL = self.a[L] * (1 - self.a[L])
        delta_L = dE_da * dSigma_zL  # dE_dZ(L)

        a_hat = torch.cat((bias, self.a[L - 1]), 0)

        self.dE_dTheta[L - 1] = torch.mm(delta_L, a_hat.t())
        #print('dE_dTheta[L-1] size is {}'.format(self.dE_dTheta[L-1]))

        delta = {}
        delta[L] = delta_L 
        for i in range(L - 1, 0, -1):
            # print(i)
            theta_back = self.Theta[self.keys[i]][:, 1: ]
            #print('i = {}, theta_back.size() is {}'.format(i, theta_back.size()))
            delta[i] = torch.mm(theta_back.t(), delta[i+1]) * (self.a[i] * (1 - self.a[i]))
            #print('a_hat is {}'.format(a_hat))
            a_hat = torch.cat((bias, self.a[i-1]), 0)
            #print('a_hat is {}'.format(a_hat))
            self.dE_dTheta[i-1] = torch.mm(delta[i], a_hat.t())
            #print('dE_dTheta is {}'.format(self.dE_dTheta))

        #print('dE_dTheta is {}'.format(self.dE_dTheta[L-1]))

    def updateParams(self, eta: float):
        for i in range(self.szLayer - 1):
            self.Theta[self.keys[i]] = self.Theta[self.keys[i]] - eta * self.dE_dTheta[i]
