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
            y = 0.1
            z = random.uniform(0, 1)

            vec1 = np.array([(-1) * np.dot(normal_vector[1:3], np.array([y, z])) / normal_vector[0], y, z])
            vec1 /= np.linalg.norm(vec1)

            y = random.uniform(0, 1)
            z = 0.1
            vec2 = np.array([(-1) * np.dot(normal_vector[1:3], np.array([y, z])) / normal_vector[0], y, z])
            vec2 -= np.dot(vec1, vec2) * vec1
            vec2 /= np.linalg.norm(vec2)
    elif normal_vector[0] == 0 and normal_vector[1] == 0:
        return np.array([1, 0, 0]), np.array([0, 1, 0])
    else:
        x = 0.1
        y = random.uniform(0, 1)

        vec1 = np.array([x, y, (-1) * np.dot(normal_vector[0:2], np.array([x, y])) / normal_vector[2]])
        vec1 /= np.linalg.norm(vec1)

        x = random.uniform(0, 1)
        y = 0.1
        vec2 = np.array([x, y, (-1) * np.dot(normal_vector[0:2], np.array([x, y])) / normal_vector[2]])
        vec2 -= np.dot(vec1, vec2) * vec1
        vec2 /= np.linalg.norm(vec2)
    return vec1, vec2


def projection(vec1, vec2, vec2_is_normal=False):
    v = np.dot(vec1, vec2) * vec2
    return v if vec2_is_normal else v / np.linalg.norm(vec2)


def oblique_MPR(volume, spacing, point, normal_vector_plane, width, output_shape, prior_vec=None):
    u, v = two_vec_on_plane(normal_vector_plane)

    width = np.array(width)
    output_shape = np.array(output_shape)
    if width.ndim == 0:
        width = np.full(2, width)
    if output_shape.ndim == 0:
        output_shape = np.full(2, output_shape)

    if prior_vec:
        v1, v2 = prior_vec
        u1 = projection(v1, u) + projection(v1, v)
        v1 = projection(v2, u) + projection(v2, v)
        u, v = u1 / np.linalg.norm(u1), v1 / np.linalg.norm(v1)
    coords = (u[:, None, None] * np.linspace(width[0] / -2.0, width[0] / 2.0, output_shape[0])[None, :, None] +
              v[:, None, None] * np.linspace(width[1] / -2.0, width[1] / 2.0, output_shape[1])[None, None, :])

    idx = coords / spacing[(slice(None),) + (None,) * (coords.ndim - 1)] + point[:, None, None]
    new_data = map_coordinates(volume, idx, order=1, mode='constant', cval=-1024)
    return new_data, u, v


def oblique_MPR_2(volume, spacing, point, normal_vector_plane, width, output_shape, prior_vec=None):
    u, v = two_vec_on_plane(normal_vector_plane)
    width = np.array(width)
    output_shape = np.array(output_shape)
    if width.ndim == 0:
        width = np.full(3, width)
    if output_shape.ndim == 0:
        output_shape = np.full(3, output_shape)

    if prior_vec:
        v1, v2 = prior_vec
        u1 = projection(v1, u) + projection(v1, v)
        v1 = projection(v2, u) + projection(v2, v)
        u, v = u1 / np.linalg.norm(u1), v1 / np.linalg.norm(v1)
    sl = (slice(None),) + (None,) * output_shape.shape[0]
    n = normal_vector_plane
    coords = (n[sl] * np.linspace(width[0] / -2.0, width[0] / 2.0, output_shape[0])[None, :, None, None] +
              u[sl] * np.linspace(width[1] / -2.0, width[1] / 2.0, output_shape[1])[None, None, :, None] +
              v[sl] * np.linspace(width[2] / -2.0, width[2] / 2.0, output_shape[2])[None, None, None, :])

    idx = coords / spacing[(slice(None),) + (None,) * (coords.ndim - 1)] + point[sl]
    new_data = map_coordinates(volume, idx, order=1, mode='constant', cval=-1024)
    return new_data, u, v


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
    ct.load_image_data(
        r'C:\Users\Administrator\Downloads\CCTA\HALF 75% 1.04s Axial 1.0 CTA-HALF CTA Sharp Cardiac LUNG 75% - 7')
    # ViewCT(ct.data)
    plane = oblique_MPR(ct.data, ct.spacing, np.floor(np.array(ct.data.shape) / 2.0),
                        np.array([1, 1, 1]), 128, (128, 128))
    ViewCT(plane.reshape((1, 128, 128)))


def demo_curved_mpr():
    ct = CtVolume()
    ct.load_image_data(
        r'C:\Users\Administrator\Downloads\CCTA\HALF 75% 1.04s Axial 1.0 CTA-HALF CTA Sharp Cardiac LUNG 75% - 7')
    centerlines = get_centerlines(r'C:\Users\Administrator\Downloads\CCTA\CCA results 75% 11 TI - 1109')
    cmpr = curved_MPR(ct, centerlines['LAD'])
    ViewCT(cmpr)


def demo_centerlines_data(save=False):
    centerlines = get_centerlines(r'C:\Users\Administrator\Downloads\CCTA\CCA results 75% 11 TI - 1109')
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

def explore_data_structure_centerlines():

    centerlines = get_centerlines(r'C:\Users\Administrator\Downloads\CCTA\CCA results 75% 11 TI - 1109')
    fig = plt.figure()
    ax = Axes3D(fig)

    for vessel, points in centerlines.items():
        point1 = points[0,:3]
        point2 = points[1,:3]
        point3 = points[2,:3]
        delta1 = np.array([points[0,3],points[0,5],points[0,7]])
        delta2 = np.array([points[0,4],points[0,6],points[0,8]])
        ax.scatter(*([point1[i] for i in range(3)]), color='black')
        ax.scatter(*([point2[i] for i in range(3)]), color='gray')
        ax.scatter(*([point3[i] for i in range(3)]), color='gray')
        ax.scatter(*([point1[i]-delta1[i] for i in range(3)]), color='red')
        ax.scatter(*([point1[i]-delta2[i] for i in range(3)]), color='green')
        break
    plt.show()


def curved_MPR(ctVolume, center_points):
    points = center_points[:, 0:3]
    dim = 60
    ret = np.full((points.shape[0], dim, dim), -1024)
    ctVolume.spacing[0] = ctVolume.spacing[0] * (-1)
    max = points.shape[0] - 1
    u, v = None, None
    for i in range(max + 1):
        point1 = np.array(points[i])[::-1]
        point1 = ctVolume.absolute_to_pixel_coord(point1, True)

        if i == 0 or i == max:
            point2 = np.array(points[1] if i == 0 else points[max - 2])[::-1]
            point2 = ctVolume.absolute_to_pixel_coord(point2, True)
            normal_vec = point1 - point2
        else:
            point0 = np.array(points[i - 1])[::-1]
            point2 = np.array(points[i + 1])[::-1]
            point0 = ctVolume.absolute_to_pixel_coord(point0, True)
            point2 = ctVolume.absolute_to_pixel_coord(point2, True)
            normal_vec = point0 - point2

        ret[i], u, v = oblique_MPR_2(ctVolume.data, ctVolume.spacing, point1,
                                     normal_vec, dim, dim, None if u is None else (u, v))

    return ret


if __name__ == '__main__':
    np.set_printoptions(formatter={'float_kind': lambda x: "%.2f" % x})
    explore_data_structure_centerlines()
    # demo_curved_mpr()
