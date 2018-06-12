import random
from collections import OrderedDict
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import map_coordinates


def get_float_index(arr, index):
    shape = np.array(arr.shape)
    index = np.array(index)
    if not (np.all(index > np.zeros(index.shape)) and np.all(index < shape)):
        return


def two_vec_on_plane(normal_vector):
    if normal_vector[2] == 0:
        if normal_vector[1] == 0:
            return np.array([0, 1, 0]), np.array([0, 0, 1])
        elif normal_vector[0] == 0:
            return np.array([1, 0, 0]), np.array([0, 0, 1])
        else:
            y = random.uniform(0, 1)
            z = random.uniform(0, 1)

            vec1 = np.array([(-1) * np.dot(normal_vector[1:3], np.array([y, z])) / normal_vector[0], y, z])
            vec1 /= np.linalg.norm(vec1)

            x = random.uniform(0, 1)
            y = random.uniform(0, 1)
            vec2 = np.array([(-1) * np.dot(normal_vector[1:3], np.array([y, z])) / normal_vector[0], y, z])
            vec2 -= np.dot(vec1, vec2) * vec1
            vec2 /= np.linalg.norm(vec2)
    elif normal_vector[0] == 0 and normal_vector[1] == 0:
        return np.array([1, 0, 0]), np.array([0, 1, 0])
    else:
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)

        vec1 = np.array([x, y, (-1) * np.dot(normal_vector[0:2], np.array([x, y])) / normal_vector[2]])
        vec1 /= np.linalg.norm(vec1)

        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        vec2 = np.array([x, y, (-1) * np.dot(normal_vector[0:2], np.array([x, y])) / normal_vector[2]])
        vec2 -= np.dot(vec1, vec2) * vec1
        vec2 /= np.linalg.norm(vec2)
    return vec1, vec2


def oblique_MPR(volume, spacing, point, normal_vector_plane, width, output_shape):
    u, v = two_vec_on_plane(normal_vector_plane)
    coords = (u[:, None, None] * np.linspace(width / -2.0, width / 2.0, output_shape[0])[None, :, None] +
              v[:, None, None] * np.linspace(width / -2.0, width / 2.0, output_shape[1])[None, None, :])

    idx = coords / spacing[(slice(None),) + (None,) * (coords.ndim - 1)] + point[:, None, None]
    new_data = map_coordinates(volume, idx, order=1, mode='constant', cval=0)


def get_centerlines(series_dir):
    ret = OrderedDict()

    for i, file in enumerate(sorted(glob(str(Path(series_dir).joinpath('*.dcm'))))):
        if i == 0:
            continue
        f = pydicom.read_file(file)
        try:
            vessel_name = f[0x00E11040].value
        except:
            continue

        try:
            center_infos = f[0x07A11012].value
        except:
            continue

        ret[vessel_name] = np.array(center_infos).reshape((-1, 9))

    return ret


if __name__ == '__main__':
    np.set_printoptions(formatter={'float_kind': lambda x: "%.2f" % x})

    data = np.random.rand(10, 10, 10)
    print(oblique_MPR(data, np.array([1, 2, 1]), np.array([5, 5, 5]), np.array([1, 0, 0]), 3, (3, 3)))

    # centerlines = get_centerlines(r'C:\Users\Administrator\Downloads\Cad\STATE0 - 805')
    centerlines = get_centerlines(r'C:\Users\Administrator\Downloads\2994481\STATE0 - 3015')

    print(centerlines['LAD'][0])

    # data = []
    # data2 = []
    # for vessel, points in centerlines.items():
    #     data += list(points[:, 0:3])
    # for i in range(3):
    #     data[i]=centerlines['LAD'][i, 0:3]
    #
    #
    # data2+=[list(centerlines['LAD'][0, 0:3]+centerlines['LAD'][0, 3:6])]
    # data2+=[list(centerlines['LAD'][1, 0:3]+centerlines['LAD'][1, 3:6])]
    # data2+=[list(centerlines['LAD'][1, 0:3]+centerlines['LAD'][1, 6:9])]

    # data = np.array(data)
    # data2 = np.array(data2)

    fig = plt.figure()
    ax = Axes3D(fig)
    for vessel, points in centerlines.items():
        data = points[:, 0:3]
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=np.random.rand(3, ))
    # ax.scatter(data2[:, 0], data2[:, 1], data2[:, 2], color='red')
    for angle in range(0, 360, 10):
        ax.view_init(30, angle)
        # plt.draw()
        plt.savefig('coronary%d.png' % (angle,))
        # plt.pause(.001)
