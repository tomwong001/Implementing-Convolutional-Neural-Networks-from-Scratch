import numpy as np
import scipy

class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self, dLdA):

        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ

        return dLdZ


class Sigmoid:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Sigmoid.
    """
    def forward(self, Z):

        self.A = 1 / (1 + np.exp(-Z))

        return self.A

    def backward(self, dLdA):

        sigmoid_derivative = self.A - self.A * self.A

        dLdZ = dLdA * sigmoid_derivative

        return dLdZ



class Tanh:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Tanh.
    """
    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dLdA):

        tanh_derivative = 1 - np.square(self.A)
        dLdZ = dLdA * tanh_derivative
        return dLdZ


class ReLU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on ReLU.
    """

    def forward(self, Z):
        self.A = np.maximum(0, Z)
        return self.A

    def backward(self, dLdA):
        dAdZ = np.where(self.A > 0, 1, 0)
        dLdZ = dLdA * dAdZ
        return dLdZ

class GELU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on GELU.
    """


    def forward(self, Z):
        self.Z = Z
        self.A = 1 / 2 * Z * (1 + scipy.special.erf(Z / np.sqrt(2)))
        return self.A

    def backward(self, dLdA):
        gelu_derivative = 1 / 2 * (1 + scipy.special.erf(self.Z / np.sqrt(2))) + (self.Z / np.sqrt(2 * np.pi)) * np.exp(-self.Z**2 / 2)
        dLdZ = dLdA * gelu_derivative
        return dLdZ

class Softmax:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Softmax.
    """

    def forward(self, Z):
        """
        Remember that Softmax does not act element-wise.
        It will use an entire row of Z to compute an output element.
        """
        A = np.zeros_like(Z)
        for i in range(Z.shape[0]):  
            exps = np.exp(Z[i])
            sum_exps = np.sum(exps)
            for j in range(Z.shape[1]):  
                A[i, j] = exps[j] / sum_exps
        self.A = A
        return self.A
    

    def backward(self, dLdA):

        # Calculate the batch size and number of features
        N = dLdA.shape[0] 
        C = dLdA.shape[1]

        # Initialize the final output dLdZ with all zeros. Refer to the writeup and think about the shape.
        dLdZ = np.zeros_like(dLdA)

        # Fill dLdZ one data point (row) at a time
        for i in range(N):

            # Initialize the Jacobian with all zeros.
            J = np.zeros((C, C))

            # Fill the Jacobian matrix according to the conditions described in the writeup
            for m in range(C):
                for n in range(C):
                    if m == n:
                        J[m,n] = self.A[i, m] * (1 - self.A[i, m])
                    else:
                        J[m, n] = -self.A[i, m] * self.A[i, n]

            # Calculate the derivative of the loss with respect to the i-th input
            dLdZ[i,:] = dLdA[i, :] @ J

        return dLdZ