import codecs
import random
from collections import OrderedDict
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import map_coordinates

from helper import ViewCT, CtVolume


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
    return new_data


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


def demo_oblique_mpr():
    ct = CtVolume()
    ct.load_image_data(r'./CCTA/HALF 75% 1.04s Axial 1.0 CTA-HALF CTA Sharp Cardiac LUNG 75% - 7')
    # ViewCT(ct.data)
    plane = oblique_MPR(ct.data, ct.spacing, np.floor(np.array(ct.data.shape)/2.0),
                        np.array([1,1,1]), 128, (128, 128))
    ViewCT(plane.reshape((1, 128, 128)))

def demo_curved_mpr():
    ct = CtVolume()
    ct.load_image_data(r'./CCTA/HALF 75% 1.04s Axial 1.0 CTA-HALF CTA Sharp Cardiac LUNG 75% - 7')
    centerlines = get_centerlines(r'./CCTA/CCA results 75% 11 TI - 1109')
    cmpr= curved_MPR(ct, centerlines['LAD'])
    ViewCT(cmpr)


def demo_centerlines_data(save=False):
    centerlines = get_centerlines(r'./CCTA/CCA results 75% 11 TI - 1109')
    fig = plt.figure()
    ax = Axes3D(fig)
    for vessel, points in centerlines.items():
        data = points[:, 0:3]
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=np.random.rand(3, ))
    # ax.scatter(data2[:, 0], data2[:, 1], data2[:, 2], color='red')
    if not save:
        plt.show()
    else:
        for angle in range(0, 360, 10):
            ax.view_init(30, angle)
            plt.savefig('coronary%d.png' % (angle,))
            # plt.pause(.001)

def curved_MPR(ctVolume, center_points):
    points = center_points[:, 0:3]
    ret = np.full((points.shape[0],30,30), -1024)
    ctVolume.spacing[0] = ctVolume.spacing[0]*(-1)
    for i in range(points.shape[0]):
        point1 = np.array(points[i])[::-1]
        point2 = np.array(points[i-1] if i>0 else points[1])[::-1]
        point1 = ctVolume.absolute_to_pixel_coord(point1, True)
        point2 = ctVolume.absolute_to_pixel_coord(point2, True)
        normal_vec = point1 - point2
        ret[i] = oblique_MPR(ctVolume.data, ctVolume.spacing, point1,
                        normal_vec, 30, (30, 30))

    return ret


def output_text(centerlines, file_name):
    dim1 = dim2 = 0
    for vessel, points in centerlines.items():
        dim1 += 1
        dim2 = max(dim2, points.shape[0])
    ret = np.zeros((dim1, dim2 + 1), dtype=object)
    ind = -1
    for vessel, points in centerlines.items():
        ind += 1
        ret[ind, 0] = vessel
        for i, point in enumerate(points):
            ret[ind, i + 1] = np.array2string(point[0:3], formatter={'float_kind': lambda x: "%.4f" % x},
                                              separator=',')[1:-1]
    file = codecs.open(file_name, "w", "utf-8")
    txt = np.savetxt(file, ret.T, delimiter='\t')
    file.close()
    return txt

if __name__ == '__main__':
    np.set_printoptions(formatter={'float_kind': lambda x: "%.3f" % x})
    # demo_curved_mpr()
    for d in Path('F:').iterdir():
        if not d.is_dir() or d.parts[-1].startswith('.'):
            continue
        for subdir in d.iterdir():
            break
        centerlines = get_centerlines(str(subdir.resolve()))
        chartNo = d.parts[-1]

        txt = output_text(centerlines, chartNo)
