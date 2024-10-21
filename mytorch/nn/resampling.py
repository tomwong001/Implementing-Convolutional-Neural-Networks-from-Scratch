import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        w_out = self.upsampling_factor * (A.shape[2] - 1) + 1
        Z = np.zeros((A.shape[0], A.shape[1], w_out), dtype=A.dtype)

        for i in range(A.shape[2]):
            Z[:, :, i * self.upsampling_factor] = A[:, :, i]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        dLdA = dLdZ[:, :, ::self.upsampling_factor]  # TODO

        return dLdA


class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        self.w_in = A.shape
        Z = A[:, :, ::self.downsampling_factor]  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        dLdA = np.zeros(self.w_in)  # TODO
        dLdA[:, :, ::self.downsampling_factor] = dLdZ

        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        w_out = self.upsampling_factor * (A.shape[3] - 1) + 1
        h_out = self.upsampling_factor * (A.shape[2] - 1) + 1

        Z = np.zeros((A.shape[0], A.shape[1], h_out, w_out), dtype=A.dtype)

        for i in range(A.shape[2]):
            for j in range(A.shape[3]):
                Z[:, :, i * self.upsampling_factor, j * self.upsampling_factor] = A[:, :, i, j]

        return Z   # TODO


    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """


        dLdA = dLdZ[:, :, ::self.upsampling_factor, ::self.upsampling_factor] # TODO

        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        self.w_in = A.shape
        Z = A[:, :, ::self.downsampling_factor, ::self.downsampling_factor]  # TODO

        return Z


    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        dLdA = np.zeros(self.w_in)  # TODO
        dLdA[:, :, ::self.downsampling_factor, ::self.downsampling_factor] = dLdZ

        return dLdA
    

