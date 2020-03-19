from neural_network import NeuralNetwork
import torch


class AND():
    def __init__(self):
        self.And = NeuralNetwork([2, 1])
        self.theta0 = self.And.getLayer(0)
        self.theta0[0, 0] = -3.
        self.theta0[1, 0] = 2.
        self.theta0[2, 0] = 2.

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


class OR():
    def __init__(self):
        self.Or = NeuralNetwork([2, 1])
        self.theta0 = self.Or.getLayer(0)
        self.theta0[0, 0] = -1.
        self.theta0[1, 0] = 2.
        self.theta0[2, 0] = 2.

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


class NOT():
    def __init__(self):
        self.Not = NeuralNetwork([1, 1])
        self.theta0 = self.Not.getLayer(0)
        self.theta0[0, 0] = 1.
        self.theta0[1, 0] = -2.

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


class XOR():
    def __init__(self):
        self.Xor = NeuralNetwork([2, 2, 1])
        self.theta0 = self.Xor.getLayer(0)
        # print(self.theta0.size())
        self.theta0[0, 0] = -10.
        self.theta0[1, 0] = 20.
        self.theta0[2, 0] = 20.
        self.theta0[0, 1] = 30.
        self.theta0[1, 1] = -20.
        self.theta0[2, 1] = -20.

        self.theta1 = self.Xor.getLayer(1)
        self.theta1[0, 0] = -30.
        self.theta1[1, 0] = 20.
        self.theta1[2, 0] = 20.

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
