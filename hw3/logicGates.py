from NeuralNetwork import NeuralNetwork
import torch


class AND():
    def __init__(self):
        self.And = NeuralNetwork([2, 1])

    def __call__(self, *arg):
        return self.forward(arg)

    def forward(self, inBools):
        if len(inBools) != 2:
            print("input parameters for AND must be 2")
            return
        else:
            # convert booleans to integers
            inVals = torch.zeros(2, 1)
            for i in range(2):
                if (inBools[i] == True):
                    inVals[i] = 1

            outVal = self.And.forward(inVals)
            if outVal < 0.5:
                return False
            else:
                return True

    def train(self):
        iters = 3000
        #in_data = torch.Tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
        # out_data = torch.Tensor([[0, 0, 0, 1]])
        # in_data = torch.Tensor([[0], [0]])
        # out_data = torch.Tensor([[0]])
        for i in range(iters):
            x1 = torch.rand(1, 1)
            x2 = torch.rand(1, 1)
            t = 0.
            if x1 <= 0.5:
                x1 = 0.
            else:
                x1 = 1.

            if x2 <= 0.5:
                x2 = 0.
            else:
                x2 = 1.

            if (x1 == 1. and x2 == 1.):
                t = 1.

            in_data = torch.FloatTensor([[x1], [x2]])
            out_data = torch.FloatTensor([[t]])
            self.And.forward(in_data)
            self.And.backward(out_data)
            self.And.updateParams(0.5)

        print(self.And.getLayer(0))


class OR():
    def __init__(self):
        self.Or = NeuralNetwork([2, 1])

    def __call__(self, *arg):
        return self.forward(arg)

    def forward(self, inBools):
        if len(inBools) != 2:
            print("input parameters for OR must be 2")
            return
        else:
            # convert booleans to integers
            inVals = torch.zeros(2, 1)
            for i in range(2):
                if (inBools[i] == True):
                    inVals[i] = 1

            outVal = self.Or.forward(inVals)
            if outVal < 0.5:
                return False
            else:
                return True

    def train(self):
        iters = 3000
        #in_data = torch.Tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
        # out_data = torch.Tensor([[0, 0, 0, 1]])
        # in_data = torch.Tensor([[0], [0]])
        # out_data = torch.Tensor([[0]])

        for i in range(iters):
            x1 = torch.rand(1, 1)
            x2 = torch.rand(1, 1)
            if x1 <= 0.5:
                x1 = 0.
            else:
                x1 = 1.

            if x2 <= 0.5:
                x2 = 0.
            else:
                x2 = 1.

            if (x1 == 0. and x2 == 0.):
                t = 0.
            else:
                t = 1.

            in_data = torch.FloatTensor([[x1], [x2]])
            out_data = torch.FloatTensor([[t]])

            self.Or.forward(in_data)
            self.Or.backward(out_data)
            self.Or.updateParams(0.5)

        print(self.Or.getLayer(0))


class NOT():
    def __init__(self):
        self.Not = NeuralNetwork([1, 1])

    def __call__(self, *arg):
        return self.forward(arg)

    def forward(self, inBools):
        if len(inBools) != 1:
            print("input parameters for NOT must be 1")
            return
        else:
            # convert booleans to integers
            inVals = torch.zeros(1, 1)
            for i in range(1):
                if (inBools[i] == True):
                    inVals[i] = 1

            outVal = self.Not.forward(inVals)
            if outVal < 0.5:
                return False
            else:
                return True

    def train(self):
        iters = 3000
        #in_data = torch.Tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
        # out_data = torch.Tensor([[0, 0, 0, 1]])
        # in_data = torch.Tensor([[0], [0]])
        # out_data = torch.Tensor([[0]])

        for i in range(iters):
            x1 = torch.rand(1, 1)
            if x1 <= 0.5:
                x1 = 0.
            else:
                x1 = 1.

            if x1 == 0:
                t = 1.
            else:
                t = 0.

            in_data = torch.FloatTensor([[x1]])
            out_data = torch.FloatTensor([[t]])
            self.Not.forward(in_data)
            self.Not.backward(out_data)
            self.Not.updateParams(0.5)

        print(self.Not.getLayer(0))


class XOR():
    def __init__(self):
        self.Xor = NeuralNetwork([2, 2, 1])

    def __call__(self, *arg):
        return self.forward(arg)

    def forward(self, inBools):
        if len(inBools) != 2:
            print("input parameters for XOR must be 2")
            return
        else:
            # convert booleans to integers
            inVals = torch.zeros(2, 1)
            for i in range(2):
                if (inBools[i] == True):
                    inVals[i] = 1

            outVal = self.Xor.forward(inVals)
            if outVal < 0.5:
                return False
            else:
                return True

    def train(self):
        iters = 200000

        for i in range(iters):
            x1 = torch.rand(1, 1)
            x2 = torch.rand(1, 1)
            if x1 <= 0.5:
                x1 = 0.
            else:
                x1 = 1.

            if x2 <= 0.5:
                x2 = 0.
            else:
                x2 = 1.

            if (x1 == 0. and x2 == 0.):
                t = 0.
            elif (x1 == 1. and x2 == 1.):
                t = 0.
            else:
                t = 1.

            in_data = torch.FloatTensor([[x1], [x2]])
            out_data = torch.FloatTensor([[t]])

            self.Xor.forward(in_data)
            self.Xor.backward(out_data)
            self.Xor.updateParams(0.5)

        print(self.Xor.getLayer(0))
        print(self.Xor.getLayer(1))
