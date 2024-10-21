import numpy as np


class Linear:

    def __init__(self, in_features, out_features, weight_init_fn, bias_init_fn, debug=False):
        """
        Initialize the weights and biases with zeros
        Checkout np.zeros function.
        Read the writeup to identify the right shapes for all.
        """
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros((out_features, 1))

        self.debug = debug

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        Read the writeup for implementation details
        """
        self.A = A
        self.N = A.shape[0]  # store the batch size of input
        # Think how will self.Ones helps in the calculations and uncomment below
        self.Ones = np.ones((self.N,1))
        Z = A.dot(self.W.T) + self.Ones.dot(self.b.T)

        return Z

    def backward(self, dLdZ):

        dLdA = dLdZ.dot(self.W)  
        self.dLdW = dLdZ.T.dot(self.A)  
        self.dLdb = dLdZ.T.dot(self.Ones)  

        if self.debug:
            
            self.dLdA = dLdA

        return dLdA
