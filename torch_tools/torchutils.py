from __future__ import division, print_function
import numpy as np


def linear_interpolation_2D(input_array, indices, outside_val=0, boundary_correction=True):
    # http://stackoverflow.com/questions/6427276/3d-interpolation-of-numpy-arrays-without-scipy

    ind_0 = indices[0, :]
    ind_1 = indices[1, :]
    print('Linear_interpolation_2D input_array.shape: ', input_array.shape)
    # N2 is the number of channels, rotation is performed per channel
    N0, N1, N2 = input_array.shape
    print('N0, N1, N2 ', N0, N1, N2, indices[0].shape)

    output = np.empty([indices[0].shape[0], N2])

    x0_0 = ind_0.astype(np.integer)
    x1_0 = ind_1.astype(np.integer)
    x0_1 = x0_0 + 1
    x1_1 = x1_0 + 1

    # Check if inds are beyond array boundary:
    if boundary_correction:
        # put all samples outside datacube to 0
        inds_out_of_range = (x0_0 < 0) | (x0_1 < 0) | (x1_0 < 0) | (x1_1 < 0) |  \
                            (x0_0 >= N0) | (x0_1 >= N0) | (x1_0 >= N1) | (x1_1 >= N1)

        x0_0[inds_out_of_range] = 0
        x1_0[inds_out_of_range] = 0
        x0_1[inds_out_of_range] = 0
        x1_1[inds_out_of_range] = 0

    w0 = ind_0 - x0_0
    w1 = ind_1 - x1_0
    # Replace by this...
    # input_array.take(np.array([x0_0, x1_0, x2_0]))
    print('output.shape ', output.shape)
    for i in range(N2):  # apply rotation per channel
        output[:, i] = (input_array[x0_0, x1_0, i] * (1 - w0) * (1 - w1)
                        + input_array[x0_1, x1_0, i] * w0 * (1 - w1)
                        + input_array[x0_0, x1_1, i] * (1 - w0) * w1
                        + input_array[x0_1, x1_1, i] * w0 * w1)
    # print('output[:, 0].shape ', output[:, 0].shape)
    if boundary_correction:
        output[inds_out_of_range, :] = 0

    return output


def random_rotation(data, width=28, height=28, channels=3):
    rot = np.random.rand() * 360  # Random rotation
    grid = getGrid([width, height])
    grid = rotate_grid_2D(grid, rot)
    grid += 13.5
    if len(data.shape) > 1:
        data = linear_interpolation_2D(data, grid)
        data = np.reshape(data, [width, height, channels])
    print('data.shape ', data.shape, 'data[0].shape', data[0].shape,
          'data[:,:,0].shape', data[:, :, 0].shape)
    data[:, :, 0] = data[:, :, 0] / float(np.max(data[:, :, 0]))
    data[:, :, 1] = data[:, :, 1] / float(np.max(data[:, :, 1]))
    data[:, :, 2] = data[:, :, 2] / float(np.max(data[:, :, 2]))
    return data.astype('float32')


def getGrid(siz):
    """ Returns grid with coordinates from -siz[0]/2 : siz[0]/2, -siz[1]/2 : siz[1]/2, ...."""
    space = [np.linspace(-(N / 2), (N / 2), N) for N in siz]
    mesh = np.meshgrid(*space, indexing='ij')
    mesh = [np.expand_dims(ax.ravel(), 0) for ax in mesh]

    return np.concatenate(mesh)


def rotate_grid_2D(grid, theta):
    """ Rotate grid """
    theta = np.deg2rad(theta)

    x0 = grid[0, :] * np.cos(theta) - grid[1, :] * np.sin(theta)
    x1 = grid[0, :] * np.sin(theta) + grid[1, :] * np.cos(theta)

    grid[0, :] = x0
    grid[1, :] = x1
    return grid
