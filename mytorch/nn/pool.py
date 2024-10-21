import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel
        self.max_indices = None

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        N, C_in, W_in, H_in = A.shape
        HK, WK = self.kernel, self.kernel
        W_out = W_in - WK + 1
        H_out = H_in - HK + 1
        Z = np.zeros((N, C_in, W_out, H_out))
        self.max_indices = np.zeros((N, C_in, W_out, H_out, 2), dtype=int)

        for n in range(N):
            for c in range(C_in):
                for w in range(W_out):
                    for h in range(H_out):
                        window = A[n, c, w:w+WK, h:h+HK]
                        max_val = np.max(window)
                        Z[n, c, w, h] = max_val
                        max_pos = np.unravel_index(np.argmax(window, axis=None), window.shape)
                        self.max_indices[n, c, w, h] = (w + max_pos[0], h + max_pos[1])
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        N, C_in, W_out, H_out = dLdZ.shape
        WK, HK = self.kernel, self.kernel
        W_in = W_out + WK - 1
        H_in = H_out + HK - 1
        dLdA = np.zeros((N, C_in, W_in, H_in))

        for n in range(N):
            for c in range(C_in):
                for w in range(W_out):
                    for h in range(H_out):
                        max_w, max_h = self.max_indices[n, c, w, h]
                        dLdA[n, c, max_w, max_h] += dLdZ[n, c, w, h]

        return dLdA


class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        N, C_in, W_in, H_in = A.shape
        WK, HK = self.kernel, self.kernel
        W_out = W_in - WK + 1
        H_out = H_in - HK + 1
        Z = np.zeros((N, C_in, W_out, H_out))

        for n in range(N):
            for c in range(C_in):
                for w in range(W_out):
                    for h in range(H_out):
                        Z[n, c, w, h] = np.mean(A[n, c, w:w+WK, h:h+HK])
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        N, C_in, W_out, H_out = dLdZ.shape
        WK, HK = self.kernel, self.kernel
        W_in = W_out + WK - 1
        H_in = H_out + HK - 1
        dLdA = np.zeros((N, C_in, W_in, H_in))

        for n in range(N):
            for c in range(C_in):
                for w in range(W_out):
                    for h in range(H_out):
                        dLdA[n, c, w:w+WK, h:h+HK] += dLdZ[n, c, w, h] / (WK * HK)

        return dLdA
    


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)  # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        max_pooled = self.maxpool2d_stride1.forward(A) 
        Z = self.downsample2d.forward(max_pooled)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdMP = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdMP)
        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)  # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        mean_pooled = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(mean_pooled)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdMP = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdMP)
        return dLdA
