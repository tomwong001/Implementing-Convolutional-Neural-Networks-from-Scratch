# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        N, C_in, W_in = A.shape
        output_size = W_in - self.kernel_size + 1
        Z = np.zeros((N, self.out_channels, output_size))

        for i in range(output_size):
            patch = A[:, :, i:i+self.kernel_size]
            Z[:, :, i] = np.tensordot(patch, self.W, axes=([1, 2], [1, 2])) + self.b

        self.A = A
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        N, c_out, w_out = dLdZ.shape
        input_size = self.A.shape[2]
        dLdA = np.zeros_like(self.A)
        self.dLdW = np.zeros_like(self.W)
        self.dLdb = np.zeros_like(self.b)

        for i in range(self.out_channels):
            self.dLdb[i] = np.sum(dLdZ[:, i, :])

        for j in range(self.out_channels):
            for k in range(self.in_channels):
                for l in range(self.kernel_size):
                    self.dLdW[j, k, l] += np.sum(dLdZ[:, j, :] * self.A[:, k, l:l + w_out])

        flipped_W = np.flip(self.W, axis=2)
        padded_dLdZ = np.pad(dLdZ, ((0, 0), (0, 0), (self.kernel_size - 1, self.kernel_size - 1)), 'constant', constant_values=0)
        for i in range(N):
            for j in range(self.in_channels):
                for k in range(input_size):
                    dLdA[i, j, k] += np.sum(padded_dLdZ[i, :, k:k + self.kernel_size] * flipped_W[:, j, :], axis=(0, 1))

        return dLdA



class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,padding = 0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride
        self.pad = padding

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)  # TODO
        self.downsample1d = Downsample1d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Pad the input appropriately using np.pad() function
        # TODO
        A_padded = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad)), mode='constant', constant_values=0)

        # Call Conv1d_stride1
        # TODO
        conv_c = self.conv1d_stride1.forward(A_padded)


        # downsample
        Z = self.downsample1d.forward(conv_c) # TODO


        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        bkwd = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(bkwd)  # TODO

        # Unpad the gradient
        dLdA = dLdA[:, :, self.pad:-self.pad]


        return dLdA
