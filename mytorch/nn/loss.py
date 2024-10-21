import numpy as np


class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N = A.shape[0]
        self.C = A.shape[1]
        se = (A - Y) ** 2
        sse = np.sum(se)
        mse = sse / (self.N * self.C)

        return mse

    def backward(self):

        dLdA = 2 * (self.A - self.Y) / (self.N * self.C)

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        N = A.shape[0]
        C = A.shape[1]
        self.N = N

        Ones_C = np.ones(C, dtype='f')
        Ones_N = np.ones(N, dtype='f')

        exp_A = np.exp(A)
        sum_exp_A = np.sum(exp_A, axis=1, keepdims=True)
        self.softmax = exp_A / sum_exp_A

        crossentropy = -Y * np.log(self.softmax)
        sum_crossentropy = np.dot(np.dot(Ones_N.T, crossentropy), Ones_C)
        L = sum_crossentropy / N

        return L

    def backward(self):
        softmax_A = np.exp(self.A) / np.sum(np.exp(self.A), axis=1, keepdims=True)
        dLdA = (softmax_A - self.Y) / self.N

        return dLdA
