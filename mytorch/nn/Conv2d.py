import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        N, C_in, H_in, W_in = A.shape
        H_out = H_in - self.kernel_size + 1
        W_out = W_in - self.kernel_size + 1
        Z = np.zeros((N, self.out_channels, H_out, W_out)) # TODO

        for n in range(N):
            for c_out in range(self.out_channels):
                for h in range(H_out):
                    for w in range(W_out):
                        patch = A[n, :, h:h+self.kernel_size, w:w+self.kernel_size]
                        Z[n, c_out, h, w] = np.sum(patch * self.W[c_out, :, :, :]) + self.b[c_out]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        N, C_out, H_out, W_out = dLdZ.shape
        N, C_in, H_in, W_in = self.A.shape
        
        self.dLdW = np.zeros_like(self.W)  # TODO
        self.dLdA = np.zeros_like(self.A)  # TODO
        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3)) # TODO
        
        pad_size = self.kernel_size - 1
        padded_dLdZ = np.pad(dLdZ, ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'constant', constant_values=(0,))
        
        flipped_W = np.flip(self.W, axis=(2, 3))
        
        for n in range(N):
            for c_in in range(C_in):
                for h in range(H_in):
                    for w in range(W_in):
                        for c_out in range(C_out):
                            dLdA_patch = padded_dLdZ[n, c_out, h:h+self.kernel_size, w:w+self.kernel_size]
                            self.dLdA[n, c_in, h, w] += np.sum(dLdA_patch * flipped_W[c_out, c_in, :, :])
        
        for o in range(self.out_channels):
            for i in range(self.in_channels):
                for k1 in range(self.kernel_size):
                    for k2 in range(self.kernel_size):
                        self.dLdW[o, i, k1, k2] += np.sum(self.A[:, i, k1:k1+H_out, k2:k2+W_out] * dLdZ[:, o, :, :])
        
        return self.dLdA




class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)  # TODO
        self.downsample2d = Downsample2d(stride)    # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """

        # Pad the input appropriately using np.pad() function
        A_padded = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), mode='constant', constant_values=0)

        # Call Conv2d_stride1
        # TODO
        conv_c = self.conv2d_stride1.forward(A_padded)

        # downsample
        Z = self.downsample2d.forward(conv_c)  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample1d backward
        # TODO
        bkwd = self.downsample2d.backward(dLdZ)

        # Call Conv2d_stride1 backward
        dLdA = self.conv2d_stride1.backward(bkwd)  # TODO

        # Unpad the gradient
        # TODO
        dLdA = dLdA[:, :, self.pad:-self.pad, self.pad:-self.pad]

        return dLdA
