# coding: utf-8

# In[1]:


# %run ../py_imports.ipynb
from numba import jit, njit

from time import time,sleep
from helper import ViewCT, CtVolume, groupedAvg
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage import map_coordinates, convolve
from scipy.sparse import coo_matrix
from scipy.optimize import fsolve
import cv2
# from skimage.measure import regionprops, label
from scipy import interpolate, optimize
from scipy.interpolate import UnivariateSpline, PPoly, CubicSpline
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import os
import re
import glob
import pydicom
import numpy as np
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import concurrent.futures
from pathlib import Path
from itertools import tee, chain
from functools import reduce
from collections import OrderedDict, namedtuple
import random
import struct
import inspect
os.environ['MKL_NUM_THREADS'] = '16'
np.set_printoptions(formatter={'float_kind': lambda x: "%.3f" % x})


# In[ ]:


def kill_thread(thread):
    import ctypes

    id = thread.ident
    code = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(id),
        ctypes.py_object(SystemError)
    )
    if code == 0:
        raise ValueError('invalid thread id')
    elif code != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(id),
            ctypes.c_long(0)
        )
        raise SystemError('PyThreadState_SetAsyncExc failed')


def kill_all_thread():
    for thread in jobs.running:
        kill_thread(thread)


# In[ ]:


def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)


# In[ ]:


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class PolyLine:
    def __init__(self, points):
        self.polys = [
            Poly(p1[:3], p2[:3], p1[3:6]) for p1, p2 in pairwise(points)
        ]

    def plot(self, n=10):
        p = reduce(lambda a, b: np.concatenate((a, b)),
                   list(map(lambda poly: poly.generate(), self.polys)))
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(*([p[:, i] for i in range(3)]), color='black')

        plt.show()


class Poly:
    def __init__(self, point1, point2, slope1):
        # point = a*t^2 + b*t + c,  0<=t<=1.0

        self.point1 = point1
        self.c = point1
        self.point2 = point2
        self.slope1 = slope1
        self.b = slope1
        self.a = point2 - point1 - slope1
        self.length = self.est_length()

    def get_value(self, t):
        return self.a * (t**2) + self.b * t + self.c

    def get_delta(self, t, d):
        return self.a * (2 * d * t + d**2) + self.b * d

    def get_deriv(self, t):
        return self.a * (2 * t) + self.b

    def est_length(self, n=201, from_t=0.0, to_t=1.0):
        d = (to_t - from_t) / (n - 1)
        l = 0
        for i in np.linspace(from_t, to_t, n):
            l += np.linalg.norm(self.get_delta(i, d))
        return l

    def generate(self, n=201, from_t=0.0, to_t=1.0):
        s = np.linspace(from_t, to_t, n)
        p = self.a[None, :] * (
            s**2)[:, None] + self.b[None, :] * s[:, None] + self.c[None, :]
        return p

    def plot(self, n=201, from_t=0.0, to_t=1.0):
        p = self.generate(n, from_t, to_t)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(*([p[:, i] for i in range(3)]), color='black')

        tangent = self.point1 + self.slope1
        ax.scatter(*([tangent[i] for i in range(3)]), 'g')

        plt.show()


def get_float_index(arr, index):
    shape = np.array(arr.shape)
    index = np.array(index)
    if not (np.all(index > np.zeros(index.shape)) and np.all(index < shape)):
        return


def two_vec_on_plane(normal_vector, known_vector_plane=None):
    '''
    create two orthogonal unit vectors, perpendicular to the normal_vector
    there are infinite pairs, specify one of the vector on the plane as known_vector_plane if needed, else random generate
    '''
    if known_vector_plane is not None:
        vec = np.cross(known_vector_plane, normal_vector)
        return known_vector_plane, vec / np.linalg.norm(vec)

    if abs(normal_vector[2]) < 1e-13:
        if abs(normal_vector[1]) < 1e-13:
            return np.array([0, 1, 0]), (np.array([0, 0, 1]) if normal_vector[0] < 0 else np.array([0, 0, -1]))
        elif abs(normal_vector[0]) < 1e-13:
            return np.array([1, 0, 0]), (np.array([0, 0, 1]) if normal_vector[0] > 0 else np.array([0, 0, -1]))(np.array([0, 0, 1]) if normal_vector[0] < 0 else np.array([0, 0, -1]))
        else:
            y = 0.1
            z = random.uniform(0, 1)

            vec1 = np.array([(-1) * np.dot(normal_vector[1:3], np.array(
                [y, z])) / normal_vector[0], y, z])
            vec1 /= np.linalg.norm(vec1)

            y = random.uniform(0, 1)
            z = 0.1
            vec2 = np.array([(-1) * np.dot(normal_vector[1:3], np.array(
                [y, z])) / normal_vector[0], y, z])
            vec2 -= np.dot(vec1, vec2) * vec1
            vec2 /= np.linalg.norm(vec2)
    elif abs(normal_vector[0]) < 1e-13 and abs(normal_vector[1]) < 1e-13:
        return np.array([1, 0, 0]), (np.array([0, 1, 0]) if normal_vector[0] < 0 else np.array([0, -1, 0]))
    else:
        x = 0.1
        y = random.uniform(0, 1)

        vec1 = np.array([
            x, y, (-1) * np.dot(normal_vector[0:2], np.array([x, y])) /
            normal_vector[2]
        ])
        vec1 /= np.linalg.norm(vec1)

        x = random.uniform(0, 1)
        y = 0.1
        vec2 = np.array([
            x, y, (-1) * np.dot(normal_vector[0:2], np.array([x, y])) /
            normal_vector[2]
        ])
        vec2 -= np.dot(vec1, vec2) * vec1
        vec2 /= np.linalg.norm(vec2)

    if np.dot(vec2, np.cross(vec1, normal_vector)) < 0:
        vec = -vec

    return vec1, vec2


def projection(vec1, vec2, vec2_is_normal=False):
    '''
    the projection of vec1 on vec2
    '''
    v = np.dot(vec1, vec2) * vec2
    return v if vec2_is_normal else v / np.dot(vec2, vec2)


def nparr(n, dim=3):
    if n is None:
        return None
    ret = np.asarray(n)
    if ret.ndim == 1 and ret.shape[0] == dim:
        return ret
    else:
        return np.repeat(n, dim)


def normalized(arr):
    return arr / np.linalg.norm(arr)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = normalized(v1)
    v2_u = normalized(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


# In[ ]:


def get_centerlines(series_dir):
    '''
    :param series_dir: path to series folder of CCA result (save file of centerlines)
    :return: OrderedDict, key: name of vessels, value: info of control points in list
    '''
    ret = OrderedDict()

    for i, file in enumerate(sorted(Path(series_dir).glob('*.dcm'))):
        #         if i == 0:
        #             continue
        f = pydicom.read_file(str(file))
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


def parse_results(series_dir_or_file):
    '''
    :param series_dir: path to series folder of CCA result (save file of centerlines)
    :return: OrderedDict, key: name of vessels, value: 
            dict with key: center_points, 
                    value: list of points
    '''
    ret = OrderedDict()

    if Path(series_dir_or_file).is_dir():
        files = sorted(Path(series_dir_or_file).glob('*.dcm'))
        if len(files)==0:
            files = sorted(Path(series_dir_or_file).glob('*'))
    elif Path(series_dir_or_file).exists():
        files = [series_dir_or_file]
    else:
        return

    for i, file in enumerate(files):

        
        try:
            f = pydicom.read_file(str(file))
            vessel_name = f[0x00E11040].value

        except:
            continue

        try:
            ret[vessel_name] = {}
            ret[vessel_name]['center_points'] = np.array(
                f[0x07A11012].value).reshape((-1, 3, 3))
        except:
            continue

#         try:
#             info1 = f[0x07A11010].value
#         except:
#             pass

        try:  # inner wall
            ret[vessel_name]['wall_infos'] = np.array(
                f[0x07A1101C].value).reshape((-1, 8, 3))
        except:
            pass

        try:  # inner wall point count
            v = f[0x01F51011].value
            ret[vessel_name]['inner_wall_point_count'] = np.array(v)
        except:
            pass
        try:  # inner wall, relative to center points
            v = f[0x01F51012].value
            ret[vessel_name]['inner_wall'] = np.array(
                struct.unpack("f"*(len(v)//4), v)).reshape(-1, 2)
        except:
            pass

#         try: # outer wall
#             ret[vessel_name]['wall_infos'] = np.array(
#                 f[0x07A11013].value).reshape((-1, 8, 3))
#         except:
#             pass
        
        try:  # outer wall point count
            v = f[0x01F51019].value
            ret[vessel_name]['outer_wall_point_count'] = np.array(v)
        except:
            pass
        try:  # outer wall, relative to center points
            v = f[0x01F51020].value
            ret[vessel_name]['outer_wall'] = np.array(
                struct.unpack("f"*(len(v)//4), v)).reshape(-1, 2)
        except:
            pass


#         ret[vessel_name] = {'wall_infos': np.array(wall_infos),
#                             'center_points': np.array(center_infos).reshape((-1, 9))}

    return ret


def demo_oblique_mpr():
    ct = CtVolume()
    ct.load_image_data(
        r'./CCTA/HALF 75% 1.04s Axial 1.0 CTA-HALF CTA Sharp Cardiac LUNG 75% - 7'
    )
    # ViewCT(ct.data)
    plane = oblique_MPR(ct.data, ct.spacing,
                        np.array(ct.data.shape) / 2.0, np.array([1, 1, 1]),
                        (128, 128))[0]
    ViewCT(plane.reshape((1, 128, 128)))


def rotate_volume(volume, angle=0.0, padding=None):
    vec = np.array([0, np.cos(angle), np.sin(angle)])
    if padding is None:
        padding = np.amin(volume)
    return oblique_MPR(
        volume=volume,
        spacing=np.array([1, 1, 1]),
        point=(np.array(volume.shape) / 2.0),
        normal_vector_plane=vec,
        output_shape=volume.shape[0:2],
        known_vector_plane=np.array([1, 0, 0]), padding=padding)[0]


def batch_rotate(volume, s=None, e=None, n=None):
    if s is None:
        s = 0.0

    if n is None:
        n = 20

    if e is None:
        e = s + 360 - 360 / n

    s, e = np.deg2rad(s), np.deg2rad(e)
    padding = np.amin(volume)
    ret = np.array(
        [rotate_volume(volume, angle, padding) for angle in np.linspace(s, e, n)])
    return ret


def demo_curved_mpr():
    ct = CtVolume()
    ct.load_image_data(
        r'./CCTA/HALF 75% 1.04s Axial 1.0 CTA-HALF CTA Sharp Cardiac LUNG 75% - 7'
    )
    centerlines = get_centerlines(r'./CCTA/CCA results 75% 11 TI - 1109')
    cmpr = curved_MPR(ct, centerlines['LAD'])
    # ViewCT(cmpr)

    # plane, _, _ = oblique_MPR(cmpr, np.array([1, 1, 1]), np.floor(np.array(cmpr.shape) / 2.0),
    #                           np.array([0, 1, 0]), 128, (128, 128))
    # ViewCT(plane.reshape((1, 128, 128)))

    batch = batch_rotate(cmpr)
    # for i in range(batch.shape[0]):
    #     im = Image.fromarray(np.interp(batch[i], (220-740, 220+740), (0.0, 255.0)).astype(np.uint8))
    #     im.save('output'+str(i+1)+'.png')

    ViewCT(batch)


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
            plt.savefig('coronary%d.png' % (angle, ))
            # plt.pause(.001)


def demo_centerline_bspline3d(artery='LAD'):
    centerlines = get_centerlines(r'./CCTA/CCA results 75% 11 TI - 1109')
    spl = BSpline3D(centerlines[artery][:, 0:3])
    spl.plot()


def explore_data_structure_centerlines():
    centerlines = get_centerlines(r'./CCTA/CCA results 75% 11 TI - 1109')
    # centerlines = get_centerlines(r'C:\Users\Administrator\Downloads\CCTA2\STATE5 - 810')
    fig = plt.figure()
    ax = Axes3D(fig)

    for vessel, points in centerlines.items():
        for i in range(10):
            ax.scatter(*([points[i, j] for j in range(3)]), color='black')
        for i in range(3):
            # for i in range(points.shape[0]-1, -1, -1):
            ax.scatter(
                *([points[i, j] + points[i, j + 3] for j in range(3)]),
                color='red')
            # ax.scatter(*([points[i,j]+points[i,j+6] for j in range(3)]), color='green')
        break
        point1 = points[0, :3]
        point2 = points[1, :3]
        # point3 = points[2,:3]
        delta1 = points[0, 3:6]
        delta2 = points[0, 6:9]
        ax.scatter(*([point1[i] for i in range(3)]), color='black')
        ax.scatter(*([point2[i] for i in range(3)]), color='gray')
        # ax.scatter(*([point3[i] for i in range(3)]), color='gray')
        ax.scatter(*([point1[i] - delta1[i] for i in range(3)]), color='red')
        ax.scatter(*([point1[i] - delta2[i] for i in range(3)]), color='green')
        break
    plt.show()


def plot_points_plane(points_plane):
    fig = plt.figure()
    ax = Axes3D(fig)
    points_plane = points_plane[:30, :, :]

    ax.scatter(*([points_plane[:, 0, i] for i in range(3)]), color='black')
    ax.plot(*([points_plane[:, 0, i] for i in range(3)]), color='black')
    ax.scatter(
        *([points_plane[:, 0, i] + points_plane[:, 1, i] for i in range(3)]),
        color='green')
    ax.scatter(
        *([points_plane[:, 0, i] + points_plane[:, 2, i] for i in range(3)]),
        color='red')

    plt.show()


# In[ ]:


# %%pixie_debugger
def demo_oblique_mpr_concept():
    spacing = np.array([2, 1, 1])
    plane, u, v = oblique_MPR_2(
        None,
        spacing=spacing,
        point=[0, 0, 0],
        normal_vector_plane=[1, 1, 0],
        output_shape=[3, 3, 3],
        prior_vec=None,
        known_vector_plane=[0, 0, 1],
        sample_spacing=[1, 1, 1])
    # plane.reshape(3,-1)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_aspect('equal')
    ax.scatter(*plane.reshape(3, -1))
    ax.add_collection3d(
        Line3DCollection([[[0, 0, 0], u / spacing], [[0, 0, 0], v / spacing]],
                         colors=['r', 'b']))
    ax.set_xlim((-2, 2))
    ax.set_ylim((-2, 2))
    ax.set_zlim((-2, 2))
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    set_axes_equal(ax)
    plt.show()


# In[ ]:


def oblique_MPR(volume,
                spacing,
                point,
                normal_vector_plane,
                output_shape,
                prior_vec=None,
                known_vector_plane=None,
                sample_spacing=None,
                padding=-2000):
    '''
    :param volume: volume data
    :param spacing: real world spacing between elements in data matrix
    :param point: 
    :param normal_vector_plane: normal vector of output plane / tangential of the point on curve
    :param output_shape:
    :param prior_vec: prior orthogonal vector pair (both on previous plane / perpendicular to previous normal vector)
    :param known_vector_plane:
    :return: 2D plane, with slice thickness = normal_v_len = spacing of other two axes
    '''

    normal_vector_plane = nparr(normal_vector_plane)
    point = nparr(point)
    spacing = nparr(spacing)
    known_vector_plane = nparr(known_vector_plane)

    u, v = two_vec_on_plane(normal_vector_plane, known_vector_plane)
    normal_v_len = np.linalg.norm(normal_vector_plane)
    if sample_spacing is None:
        sample_spacing = normal_v_len
    sample_spacing = nparr(sample_spacing, 2)
    output_shape = nparr(output_shape, 2)

    output_range = (output_shape - 1) * sample_spacing

    if prior_vec:
        u0, v0 = prior_vec
        u1 = normalized(
            projection(normalized(u0), normalized(normal_vector_plane)) - u0)

        v1 = normalized(np.cross(u1, normalized(normal_vector_plane)))

        #         v2 = normalized(projection(normalized(v0), normalized(normal_vector_plane))-v0 )

        #         u2 = np.cross(v2, normalized(normal_vector_plane))

        #         if np.dot(u1,u0)<0:
        #             u1=-u1
        #         if np.dot(u2,u0)<0:
        #             u2=-u2
        #         if np.dot(v1,u0)<0:
        #             v1=-v1
        #         if np.dot(v2,u0)<0:
        #             v2=-v2
        #         u1 = projection(u0, u) + projection(u0, v)
        #         v1 = np.cross(normalized(u1),normalized(normal_vector_plane))
        #         if np.dot(v1, v0)<0:
        #             v1=-v1
        #         u1, v1 = normalized(u1), normalized(v1)

        #         v2 = projection(v0, u) + projection(v0, v)
        #         u2 = np.cross(normalized(v2),normalized(normal_vector_plane))
        #         if np.dot(u2, u0)<0:
        #             u2=-u2
        #         u2, v2 = normalized(u2), normalized(v2)

        #         u, v = (u1+u2)/2, (v1+v2)/2
        #         v = np.cross(normalized(u0), normalized(normal_vector_plane))
        #         u = np.cross(v, normalized(normal_vector_plane))
        u, v = u1, v1
        if np.dot(u0, u) < 0:
            u = -u
        if np.dot(v0, v) < 0:
            v = -v


#     print(np.dot(u,v))

    sl = (slice(None), ) + (None, ) * output_shape.shape[0]

    #     n = normal_vector_plane/normal_v_len

    # world coords relative to point
    # using broadcasting of array shape: (3, output_shape[0], 1) + (3, 1, output_shape[1]) = (3, output_shape[0], output_shape[1])

    coords = (
        u[sl] * np.linspace(output_range[0] / -2.0, output_range[0] / 2.0,
                            output_shape[0])[None, :, None] +
        v[sl] * np.linspace(output_range[1] / -2.0, output_range[1] / 2.0,
                            output_shape[1])[None, None, :])

    # index that about to map onto original volume
    # using broadcasting of array shape: (3, output_shape[0], output_shape[1]) + (3, 1, 1)
    idx = coords / spacing[(slice(None), ) +
                           (None, ) * (coords.ndim - 1)] + point[sl]

    if volume is None:
        return idx, u, v
    else:
        new_data = map_coordinates(
            volume, idx, order=1, mode='constant', cval=padding)
        return new_data, u, v


def oblique_MPR_2(volume,
                  spacing,
                  point,
                  normal_vector_plane,
                  output_shape,
                  prior_vec=None,
                  known_vector_plane=None,
                  end_normal_vec=None,
                  sample_spacing=None,
                  padding=-2000):
    '''
    :param volume: volume data
    :param spacing: real world spacing between elements in data matrix
    :param point: 
    :param normal_vector_plane: normal vector of output plane / tangential of the point on curve
    :param output_shape:
    :param prior_vec: prior orthogonal vector pair (both on previous plane / perpendicular to previous normal vector)
    :param known_vector_plane:
    :param end_normal_vec: prior normal vector / tangential of the point
    :return:
    '''
    point = nparr(point)
    spacing = nparr(spacing)
    normal_vector_plane = nparr(normal_vector_plane)
    known_vector_plane = nparr(known_vector_plane)

    u, v = two_vec_on_plane(normal_vector_plane, known_vector_plane)
    normal_v_len = np.linalg.norm(normal_vector_plane)
    if sample_spacing is None:
        sample_spacing = normal_v_len
    sample_spacing = nparr(sample_spacing)
    output_shape = nparr(output_shape)
    output_range = (output_shape - 1) * sample_spacing

    if prior_vec:
        u0, v0 = prior_vec
        u1 = projection(u0, u) + projection(u0, v)
        v1 = np.cross(u1, normal_vector_plane)
        if np.dot(v1, v0) < 0:
            v1 = -v1
        u1, v1 = normalized(u1), normalized(v1)

        v2 = projection(v0, u) + projection(v0, v)
        u2 = np.cross(v2, normal_vector_plane)
        if np.dot(u2, u0) < 0:
            u2 = -u2
        u2, v2 = normalized(u2), normalized(v2)

        u, v = (u1 + u2) / 2, (v1 + v2) / 2

    sl = (slice(None), ) + (None, ) * output_shape.shape[0]
    n = normal_vector_plane / normal_v_len

    #     if end_normal_vec is None:
    if True:
        # world coords relative to point
        #         coords = (n[sl] * np.linspace(0.0, output_range[0]/(output_shape[0]-1),
        coords = (
            n[sl] * np.linspace(output_range[0] / -2.0, output_range[0] / 2.0,
                                output_shape[0])[None, :, None, None] +
            u[sl] * np.linspace(output_range[1] / -2.0, output_range[1] / 2.0,
                                output_shape[1])[None, None, :, None] +
            v[sl] * np.linspace(output_range[2] / -2.0, output_range[1] / 2.0,
                                output_shape[2])[None, None, None, :])

    else:  # not finished, do not use
        end_normal_vec = normalized(end_normal_vec)
        normal_vector_plane = normalized(normal_vector_plane)
        diff_vec = (end_normal_vec - normal_vector_plane) / output_shape[0]
        diff_idx = diff_vec / spacing

        seg_normal_v_len = normal_v_len / output_shape[0]
        seg_normal_vec = normal_vector_plane
        seg_point = point
        coords = np.empty((3, ) + tuple(output_shape))
        for slice_no in range(output_shape[0]):
            plane, u, v = oblique_MPR(
                volume=volume,
                spacing=spacing,
                point=seg_point,
                normal_vector_plane=normalized(seg_normal_vec) *
                seg_normal_v_len,
                output_shape=output_shape[1:3],
                prior_vec=(u, v),
                known_vector_plane=None,
                sample_spacing=normal_v_len)
            coords[:, slice_no, ...] = plane
            seg_point += diff_idx
            seg_normal_vec += diff_vec

    # index that about to map onto original volume
    idx = coords / spacing[(slice(None), ) +
                           (None, ) * (coords.ndim - 1)] + point[sl]

    if volume is None:
        return idx, u, v
    else:
        new_data = map_coordinates(
            volume, idx, order=1, mode='constant', cval=padding)
        return new_data, u, v


# In[ ]:


def curved_MPR_concept(ctVolume,
                       center_points,
                       show_range=(0.0, 1.0),
                       output_dim=(1000, 100, 100),
                       each_slices=1):
    center_points_coords = np.flip(center_points[:, 0:3], axis=1)
    center_points_tangent = np.flip(center_points[:, 3:6], axis=1)
    ret = np.empty((center_points_coords.shape[0], output_dim[1],
                    output_dim[2]))
    spacing = ctVolume.spacing
    u, v = None, None
    for i, ((p1, p2), n) in enumerate(
            zip(pairwise(center_points_coords), center_points_tangent[0:-1])):
        point_ind = ctVolume.absolute_to_pixel_coord(p1, True)
        normal_vec = normalized(n) * np.linalg.norm(p2 - p1)
        plane, u, v = oblique_MPR(
            volume=ctVolume.data,
            spacing=spacing,
            point=point_ind,
            normal_vector_plane=normal_vec,
            output_shape=output_dim[1:3],
            prior_vec=None if u is None else (u, v))
        ret[i, ...] = plane
    return ret


def curved_MPR(ctVolume, coronary, show_range=(0.0, 1.0), output_dim=(100, 100), output_spacing=None, upsample_z=1):
    data, spacing = ctVolume.data, ctVolume.spacing
    output_dim = nparr(output_dim, 2)

    center_points_diff = np.mean(
        list(map(np.linalg.norm, np.diff(coronary.center_points[:, 0], axis=0))))
    
    point_count = coronary.center_line.u.shape[0]
        
    ret = np.empty((point_count + (point_count-1)*(upsample_z-1), output_dim[0], output_dim[1]))

    min_value = np.amin(ctVolume.data)

    if output_spacing is None:
        output_spacing = center_points_diff
    output_z_spacing = center_points_diff / upsample_z
    
    vessel_ps = []
    for i, vessel_p in enumerate(coronary.upsample_sections(upsample_z)):
        vessel_ps.append(vessel_p)
        
        center_coord = vessel_p.center
        center_tangent = vessel_p.vec_z
        center_ortho = vessel_p.vec_y
        point_ind = ctVolume.absolute_to_pixel_coord(
            center_coord, return_float=True)

        plane, _, _ = oblique_MPR(
            volume=data,
            spacing=spacing,
            point=point_ind,
            normal_vector_plane=center_tangent*output_z_spacing,
            output_shape=output_dim,
            prior_vec=None,
            sample_spacing=output_spacing,
            known_vector_plane=(center_ortho),
            padding=min_value)

        ret[i, ...] = plane
    return ret, center_points_diff, vessel_ps


def curved_MPR_BS(ctVolume,
                  center_points,
                  show_range=(0.0, 1.0),
                  output_dim=(100, 100),
                  output_spacing=(0.3, 0.3, 0.3),
                  each_slices=1,
                  bs_settings={}):
    '''
    ctVolume: CtVolume class, or object with .data = numpy volume array and .spacing = zyx pixel spacing
    center_points: one vessel
    show_range: default from root (0.0) to end (1.0)
    output_dim: first dimension = along vessel; output isotropic volume; pixel spacing determined by first dimension
    '''

    center_points_coords = np.flip(
        center_points.reshape(-1, 9)[:, 0:3], axis=1)
    center_points_tangent = np.flip(
        center_points.reshape(-1, 9)[:, 3:6], axis=1)
    spacing = ctVolume.spacing
    output_dim = nparr(output_dim, 2)
    output_spacing = nparr(output_spacing)

    u, v = None, None

    spl = BSpline3D(
        np.concatenate((center_points_coords, center_points_tangent), axis=-1),
        **bs_settings)

    s, e = show_range

    output_dim = np.insert(output_dim, 0,
                           spl.total_len * (e - s) / output_spacing[0])
    split_n = output_dim[0] + 1
    points, spl_u = spl.linvalue(s=s, e=e, n=split_n, return_u=True)
    vector1 = [spl.vector1_spline(i) for i in spl_u]
    #     uv_unit_len = spl.total_len * (e-s) / (split_n - 1)

    each_slices = int(each_slices)
    assert each_slices >= 1

    output_dim[0] *= each_slices
    #     ret = np.full(output_dim, -2048)
    ret = np.empty(output_dim)

    #     first_drv = []
    #     second_drv = []
    #     last_fd = None
    # points_plane = np.zeros((points.shape[0],3,3))
    #     lines1 = []
    #     lines2 = []
    #     lines3 = []
    prior_normal_vec = None

    angle_threshold = 2

    angle_between_normal_vec = 0
    for i, ((point1, point2), vector_u, spl_u1) in enumerate(
            zip(pairwise(points), vector1[0:-1], spl_u[0:-1])):

        #         first_drv.append(normalized(spl.drv(spl_u1)))
        #         if last_fd is not None:
        #             second_drv.append(normalized(point1-last_fd))
        #         last_fd = point1

        # points_plane[i,0] = np.array(points[i])[::-1]

        #         normal_vec = point2-point1
        #         normal_vec = spl.smooth_derivative(spl_u1, nearest=0.05, samples=5)
        seg_len = np.linalg.norm(point2 - point1)
        #         print(seg_len)
        normal_vec = (normalized(point2 - point1) * seg_len)
        if prior_normal_vec is not None:
            angle_between_normal_vec = np.rad2deg(
                angle_between(prior_normal_vec, normal_vec))
        normal_vec = (normalized(point2 - point1) * seg_len)
        if prior_normal_vec is not None:
            angle_between_normal_vec = np.rad2deg(
                angle_between(prior_normal_vec, normal_vec))
#         normal_vec = normalized(spl.drv1(spl_u1))*seg_len

#         point_ind = ctVolume.absolute_to_pixel_coord(point1, True)
        point_ind = ctVolume.absolute_to_pixel_coord(point1, True)

        vector_u = normalized(vector_u)

        # if i == 0 or i == max:
        #     point2 = np.array(points[1] if i == 0 else points[max - 2])[::-1]
        #     point2 = ctVolume.absolute_to_pixel_coord(point2, True)
        #     normal_vec = point1 - point2
        # else:
        #     point0 = np.array(points[i - 1])[::-1]
        #     point2 = np.array(points[i + 1])[::-1]
        #     point0 = ctVolume.absolute_to_pixel_coord(point0, True)
        #     point2 = ctVolume.absolute_to_pixel_coord(point2, True)
        #     normal_vec = point0 - point2

        #         if each_slices==1:

        #         if angle_between_normal_vec<angle_threshold:
        #         if i<2:
        if 1:

            plane, u, v = oblique_MPR(
                volume=ctVolume.data,
                spacing=spacing,
                point=point_ind,
                normal_vector_plane=normal_vec,
                output_shape=output_dim[1:3],
                prior_vec=None if u is None else (u, v),
                sample_spacing=output_spacing[1:3],
                known_vector_plane=vector_u + point1)

            #         u = spl.drv2(spl_u1)
            #         u -= projection(u, normal_vec)
            #         u = normalized(u)

            #         plane, _, v = oblique_MPR(ctVolume.data, spacing, point1_ind,
            #                                   normal_vec, output_dim[1:3],
            #                                   None, u)

            prior_normal_vec = normal_vec
            ret[i, ...] = plane
        else:
            print(angle_between_normal_vec)
            subsample_n = int(
                np.floor(angle_between_normal_vec / angle_threshold)) + 2
            subsample = np.empty((subsample_n - 1, output_dim[1],
                                  output_dim[2]))

            for sub_i, (ii, jj) in enumerate(
                    pairwise(np.linspace(-0.5, 0.5, subsample_n))):
                point1_ = (point1 + point2) / 2 + ii * (point2 - point1)
                point2_ = (point1 + point2) / 2 + jj * (point2 - point1)
                spl_u_ = spl_u1 + ii * (spl_u1 - spl_u[i - 1])
                normal_vec_ = normalized(normal_vec + ii *
                                         (normal_vec - prior_normal_vec)
                                         ) * seg_len / (subsample_n - 1) / 2

                plane, u, v = oblique_MPR(
                    volume=ctVolume.data,
                    spacing=spacing,
                    point=ct.absolute_to_pixel_coord((point1_ + point2_) / 2,
                                                     True),
                    normal_vector_plane=normal_vec_,
                    output_shape=output_dim[1:3],
                    prior_vec=None if u is None else (u, v),
                    sample_spacing=output_spacing[1:3],
                    known_vector_plane=vector_u + point1)

                subsample[sub_i, ...] = plane

            prior_normal_vec = normal_vec_
            ret[i, ...] = np.mean(subsample, axis=0)

#         else:
#             plane, u, v = oblique_MPR_2(volume=ctVolume.data,
#                                         spacing=spacing,
#                                         point=point_ind,
#                                         normal_vector_plane=normal_vec,
#                                         output_shape=[each_slices, output_dim[1], output_dim[2]],
#                                         prior_vec=None if u is None else (u, v),
#                                         end_normal_vec = spl_drv2,
#                                         sample_spacing=(normal_vec, uv_unit_len, uv_unit_len))
#             ret[(i*each_slices):((i+1)*each_slices), ...] = plane

#         first_drv.append(spl.drv1(spl_u1))
#         second_drv.append(spl.drv2(spl_u1))
#         lines1.append([(point1), (point1 + u * 0.3)])
#         lines2.append([(point1), (point1 + v * 0.3)])
#         lines3.append([(point1), (point1 + normal_vec / seg_len * 0.3)])

#         break
# points_plane[i,1]=u
# points_plane[i,2]=v

# plot_points_plane(points_plane)

#     plt.plot([np.dot(a,b) for a,b in pairwise(second_drv)])
    return ret, spl


#     return ret, spl, np.array(lines1), np.array(lines2), np.array(lines3), np.array(first_drv), np.array(second_drv)


# In[ ]:


def running_mean(x, N):
    z = np.zeros(x[0].shape)
    cumsum = np.cumsum(np.insert(x, 0, z, axis=0), axis=0)
    return (cumsum[N:] - cumsum[:-N]) / float(N)


# In[ ]:


vessel_p = namedtuple(
    'vessel_p', 'center vec_z vec_y vec_x inner_wall outer_wall')


def coord_rel_to_abs(coords, center, vec_y, vec_x, scale=1.0):
    return np.matmul(coords, np.linalg.pinv(
        np.array([vec_y, vec_x])).T) * scale + center


def coord_abs_to_rel(coords, center, vec_y, vec_x, scale=1.0):
    return np.matmul(coords - center, np.array([vec_y, vec_x]).T) / scale

def counterclockwise_order(points, vec_z):
        vec1 = points[1]-points[0]
        vec2 = points[2] - points[1]
        return np.dot(np.cross(vec1,vec2), vec_z)>0

def lazy_property(fn):
    '''Decorator that makes a property lazy-evaluated.
    '''
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazy_property


class Coronary:
    def __init__(self, result, n_per_section=100, flip_xz=True, cp_offset=(0, 0, 0), fix_orifice=None):
        center_points = result['center_points'].copy()
        inner_wall_rel_coords = result['inner_wall'].copy()
        outer_wall_rel_coords = result['outer_wall'].copy()

#         self.oct_section_points = oct_section_points.ravel().reshape(-1, 8, 3)
        self.center_points = center_points.reshape(-1, 3, 3)
        self.inner_wall_rel_coords = inner_wall_rel_coords
        self.outer_wall_rel_coords = outer_wall_rel_coords
        
        self.cp_offset = cp_offset
        
        if flip_xz:
            self.flip_xz=True
            #             self.oct_section_points = np.flip(self.oct_section_points, axis=2)
            self.inner_wall_rel_coords = np.flip(
                self.inner_wall_rel_coords, axis=1)
            self.outer_wall_rel_coords = np.flip(
                self.outer_wall_rel_coords, axis=1)
            self.center_points = np.flip(self.center_points, axis=2)
        else:
            self.flip_xz=False
            
        self.center_points[:, 0, :] += nparr(cp_offset)

        # normal vector from center_points is not extactly on oct_ponits' plane
#         if fix:
#             for i, (oct_p, cps) in enumerate(zip(self.oct_section_points, self.center_points)):
#                 vec1 = oct_p[0]-oct_p[1]
#                 vec2 = oct_p[1]-oct_p[2]
#                 fixed_tangent = normalized(np.cross(vec1, vec2))

#                 self.center_points[i, 1, :] = fixed_tangent if np.dot(
#                     fixed_tangent, cps[1]) > 0 else -fixed_tangent

#                 fixed_normal = normalized(
#                     cps[2] - projection(cps[2], fixed_tangent))
#                 self.center_points[i, 2, :] = fixed_normal

        
    
        self.convertInnerWallPoints(result)

        if fix_orifice:
            self.fix_orifice(fix_orifice)

#         if n_per_section:
#             self.resample_section(n_per_section=n_per_section)

        self.center_line = BSpline3D(self.center_points.reshape(-1, 9))
        self.total_len = self.center_line.total_len
#         self.center_points = center_points
    def fix_orifice(self, all_results):
        cp_LAD = all_results['LAD']['center_points'].copy()
        cp_RCA = all_results['RCA']['center_points'].copy()
        
        cp_LAD=cp_LAD.reshape(-1, 3, 3)
        cp_RCA=cp_RCA.reshape(-1, 3, 3)
        
        if self.flip_xz:
            cp_LAD = np.flip(cp_LAD,axis=2)
            cp_RCA = np.flip(cp_RCA,axis=2)
        
        cp_LAD[:, 0, :] += nparr(self.cp_offset)
        cp_RCA[:, 0, :] += nparr(self.cp_offset)
        
        # sum distance for first 20 points
        n=20
        self_cp = self.center_points[0:n,0]
        ds_LAD = np.sum(np.linalg.norm(cp_LAD[0:n,0]-self_cp, axis=1))
        ds_RCA = np.sum(np.linalg.norm(cp_RCA[0:n,0]-self_cp, axis=1))
        
        orifice = cp_LAD[0] if ds_LAD<ds_RCA else cp_RCA[0]
        
        vec_z = orifice[1]
        cp = orifice[0]
        c = np.dot(vec_z, cp)
        
        sign0 = np.sign(np.dot(vec_z, self_cp[0]) - c)
        for i, p in enumerate(self.center_points[1:,0]):
            if np.sign(np.dot(vec_z,p) -c) != sign0:
                break
            if i>100:
                return
        
        i+=1
        self.center_points = self.center_points[i:]
        self.inner_wall_abs_coords = self.inner_wall_abs_coords[i:]
        self.inner_wall_rel_coords = self.inner_wall_rel_coords[i:]
        self.outer_wall_abs_coords = self.outer_wall_abs_coords[i:]
        self.outer_wall_rel_coords = self.outer_wall_rel_coords[i:]
        
            
    def center_vec(self, u):
        return
        if u == 1:
            return self.center_points[-1]

        b = np.digitize(u, self.center_line.u) - 1

        u0, u1 = self.center_line.u[b:(b+2)]
        if self.center_line.u[b] == u:
            return self.center_points[b]

        cp = self.center_line.spline(u)

        vec_z0, vec_z1 = self.center_points[b:(b+2), 1]
        vec_z0 = normalized(vec_z0)
        vec_z1 = normalized(vec_z1)

        vec_z = normalized(self.drv1(u))

        if np.dot(vec_z, ((u-u1)*vec_z0+(u-u0)*vec_z1)/(u1-u0)) < 0:
            vec_z = -vec_z

        vec_y = self.vec_y_spline(u) - cp
        vec_y = normalized(vec_y-projection(vec_y, vec_z))

#         if np.dot(vec_y, vec_y0+vec_y1)<0:
#             vec_y = -vec_y
#         vec_x = normalized(np.cross(vec_y, vec_z))
        return np.vstack((cp, vec_z, vec_y))

    def convertInnerWallPoints(self, result):
        point_count = result['inner_wall_point_count'].copy()
        points_rel_coor = self.inner_wall_rel_coords

        point_count2 = result['outer_wall_point_count'].copy()
        points_rel_coor2 = self.outer_wall_rel_coords

        ret = []
        ret2 = []
        ret3 = []
        ret4 = []
        last_ind = 0
        last_ind2 = 0
        scale = 0.5
        reverse_order = False
        
        for i, (p_count, p_count2, cp) in enumerate(zip(point_count, point_count2, self.center_points)):
            rel_coor = points_rel_coor[(last_ind):(last_ind+p_count)]
            rel_coor[:, 1] *= -1

            rel_coor2 = points_rel_coor2[(last_ind2):(last_ind2+p_count2)]
            rel_coor2[:, 1] *= -1

            center, vec_z, vec_y = cp

            vec_z = normalized(vec_z)
            vec_y = normalized(vec_y)
            vec_x = np.cross(vec_y, vec_z)
            vec_x = normalized(vec_x)
#             rel_coor = np.insert(rel_coor, 0, 0, axis=1)
#             abs_coor = np.matmul(rel_coor, np.array([vec1, vec2])) + center

            # if two succesive points are identical, spline fit show error
    
            diff_zero = np.where(
                np.max(np.abs(np.diff(rel_coor, axis=0)), axis=1) < 1e-9)[0]
            if diff_zero.size:
                rel_coor = np.delete(rel_coor, diff_zero, axis=0)
#                     print('Detect overlapping points at #{}'.format(i))

            diff_zero = np.where(
                np.max(np.abs(np.diff(rel_coor2, axis=0)), axis=1) < 1e-9)[0]
            if diff_zero.size:
                rel_coor2 = np.delete(rel_coor2, diff_zero, axis=0)
                
            abs_coor = np.matmul(rel_coor, np.linalg.pinv(
                np.array([vec_y, vec_x])).T) * scale + center
            
            if not counterclockwise_order(abs_coor, vec_z):
                abs_coor=abs_coor[::-1, ...]
                rel_coor=rel_coor[::-1, ...]
            
            ret.append(abs_coor)
            ret2.append(rel_coor)

            last_ind += p_count

            abs_coor2 = np.matmul(rel_coor2, np.linalg.pinv(
                np.array([vec_y, vec_x])).T) * scale + center
            if not counterclockwise_order(abs_coor2, vec_z):
                abs_coor2=abs_coor2[::-1, ...]
                rel_coor2=rel_coor2[::-1, ...]
            ret3.append(abs_coor2)
            ret4.append(rel_coor2)

            last_ind2 += p_count2

        self.inner_wall_abs_coords = ret
        self.inner_wall_rel_coords = ret2
        self.outer_wall_abs_coords = ret3
        self.outer_wall_rel_coords = ret4

#         self._inner_wall_abs_spl = None
#         self._inner_wall_rel_spl = None
#         self._outer_wall_abs_spl = None
#         self._outer_wall_rel_spl = None

    
        
    def resample_section(self, n_per_section=100):
        #         with concurrent.futures.ProcessPoolExecutor(
        #                 max_workers=16) as executor:
        #             self.section_spl = [
        #                 executor.submit(BSpline3D, oct_p, simple_init=True, periodic=True).result()
        #                 for oct_p in self.oct_section_points
        #             ]

        #         with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        #             self.section_points = np.array([executor.submit(spl.linvalue,n=n_per_section).result()[0:-1]
        #                                             for spl in self.section_spl])
        #         with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        #             self.longitudinal_spl = [executor.submit(BSpline3D, self.section_points[:,i,:], simple_init=True).result()
        #                                  for i in range(self.section_points.shape[1])]

        self.section_spl = [BSpline3D(
            wall_p, simple_init=True, periodic=True) for wall_p in self.inner_wall_rel_coords]
        self.section_u0 = [spl.intersect_u(p[0], p[2]) for spl, p in zip(
            self.section_spl, self.center_points)]
        self.section_points = np.array([spl.spline(np.mod(np.linspace(0, 1, n_per_section+1)+u, 1.0))[0:-1]
                                        for spl, u in zip(self.section_spl, self.section_u0)])
        self.longitudinal_spl = [BSpline3D(self.section_points[:, i, :], simple_init=True)
                                 for i in range(self.section_points.shape[1])]

    @lazy_property
    def inner_wall_rel_spl_lon(self):
        return self._wall_spl_lon(inspect.currentframe().f_code.co_name)
    @lazy_property
    def outer_wall_rel_spl_lon(self):
        return self._wall_spl_lon(inspect.currentframe().f_code.co_name)
    
    def _wall_spl_lon(self, func_name):
        in_or_out, abs_or_rel = func_name[0:5], func_name[11:14]
        ret = []
        wall_points = []
        for spl,u0 in zip(getattr(self, '{}_wall_rel_spl'.format(in_or_out)),
                          getattr(self, '{}_wall_rel_spl_u0'.format(in_or_out))):
            p=[]
            for i in np.linspace(0,1,100)[:-1]:
                p.append(np.insert(spl.spline(np.mod(i+u0,1.0)),0,i))
            wall_points.append(p)
        wall_points = np.array(wall_points)
        for i in range(wall_points.shape[1]):
            ret.append(BSpline3D(wall_points[:,i,:], simple_init=True))
        return ret
    
    @lazy_property
    def inner_wall_abs_spl(self):
        return self._wall_spl(inspect.currentframe().f_code.co_name)

    @lazy_property
    def inner_wall_rel_spl(self):
        return self._wall_spl(inspect.currentframe().f_code.co_name)

    @lazy_property
    def outer_wall_abs_spl(self):
        return self._wall_spl(inspect.currentframe().f_code.co_name)

    @lazy_property
    def outer_wall_rel_spl(self):
        return self._wall_spl(inspect.currentframe().f_code.co_name)
    
    @property
    def inner_wall_abs_spl_u0(self):
        return self._wall_spl_u0(inspect.currentframe().f_code.co_name)
    @property
    def inner_wall_rel_spl_u0(self):
        return self._wall_spl_u0(inspect.currentframe().f_code.co_name)
    @property
    def outer_wall_abs_spl_u0(self):
        return self._wall_spl_u0(inspect.currentframe().f_code.co_name)
    @property
    def outer_wall_rel_spl_u0(self):
        return self._wall_spl_u0(inspect.currentframe().f_code.co_name)

    def _wall_spl(self, func_name):
        in_or_out, abs_or_rel = func_name[0:5], func_name[11:14]
        coords = getattr(
            self, '{}_wall_{}_coords'.format(in_or_out, abs_or_rel))
        ret = []
        ret2 = []
        for wall_p, cp in zip(coords, self.center_points):
            spl = BSpline3D(wall_p, simple_init=True, periodic=True)
            ret.append(spl)
            if abs_or_rel == 'abs':
                ret2.append(spl.intersect_u(cp[0], cp[2]))
            else:
                ret2.append(spl.intersect_u(np.array([0,0]), np.array([0,1])))
        setattr(self, '_{}_wall_{}_spl_u0'.format(in_or_out, abs_or_rel), ret2)
        return ret
    
    def _wall_spl_u0(self, func_name):
        in_or_out, abs_or_rel = func_name[0:5], func_name[11:14]
        if not hasattr(self, '_{}_wall_{}_spl_u0'.format(in_or_out, abs_or_rel)):
            getattr(self, '{}_wall_{}_spl'.format(in_or_out, abs_or_rel))
        return getattr(self, '_{}_wall_{}_spl_u0'.format(in_or_out, abs_or_rel))
        

    def upsample_sections(self, upsample_ratio=1, xy_samples=100):

        for i, ((u0, u1), 
                (cp0, cp1), 
                (in0, in1), 
                (ou0, ou1), 
                (rel_in0, rel_in1), 
                (rel_ou0, rel_ou1), 
                (in_u0, in_u1), 
                (ou_u0, ou_u1)) in enumerate(zip(pairwise(self.center_line.u), 
                                                   pairwise(self.center_points),
                                                   pairwise(self.inner_wall_abs_spl),
                                                   pairwise(self.outer_wall_abs_spl),
                                                   pairwise(self.inner_wall_rel_spl),
                                                   pairwise(self.outer_wall_rel_spl),
                                                   pairwise(self.inner_wall_rel_spl_u0), 
                                                   pairwise(self.outer_wall_rel_spl_u0))):
            center0, vec_z0, vec_y0 = cp0
            vec_z0 = normalized(vec_z0)
            vec_y0 = normalized(vec_y0)
            vec_x0 = normalized(np.cross(vec_y0, vec_z0))
            
            rel_coord_in0 = rel_in0.spline(np.mod(np.linspace(0,1,xy_samples)[:-1] + in_u0,1.0)) 
            abs_coord_in0 = coord_rel_to_abs(rel_coord_in0, center0, vec_y0, vec_x0, scale=0.5)
            rel_coord_ou0 = rel_ou0.spline(np.mod(np.linspace(0,1,xy_samples)[:-1] + ou_u0,1.0)) 
            abs_coord_ou0 = coord_rel_to_abs(rel_coord_ou0, center0, vec_y0, vec_x0, scale=0.5)
            
            
            yield vessel_p(center0, vec_z0, vec_y0, vec_x0, BSpline3D(abs_coord_in0, simple_init=True, periodic=True), 
                              BSpline3D(abs_coord_ou0, simple_init=True, periodic=True))

            center1, vec_z1, vec_y1 = cp1
            vec_z1 = normalized(vec_z1)
            vec_y1 = normalized(vec_y1)
            vec_x1 = normalized(np.cross(vec_y1, vec_z1))
            
            rel_coord_in1 = rel_in1.spline(np.mod(np.linspace(0,1,xy_samples)[:-1] + in_u1,1.0)) 
            rel_coord_ou1 = rel_ou1.spline(np.mod(np.linspace(0,1,xy_samples)[:-1] + ou_u1,1.0)) 

            for j in range(1, upsample_ratio):
                u = (j*u1+(upsample_ratio-j)*u0)/upsample_ratio

                cp = self.center_line.spline(u)
#                 cp = ((u1-u)*center0+(u-u0)*center1)/(u1-u0)

#                 vec_z = normalized(self.center_line.drv1(u))

#                 if np.dot(vec_z, ((u-u1)*vec_z0+(u-u0)*vec_z1)/(u1-u0)) < 0:
#                     vec_z *= -1
#                 vec_z = normalized(((u1-u)*vec_z0+(u-u0)*vec_z1)/(u1-u0))
                vec_z = normalized(self.center_line.vec_z_spline(u))

#                 vec_y = self.center_line.vec_y_spline(u) - cp

#                 vec_y_ = normalized(((u1-u)*vec_y0+(u-u0)*vec_y1)/(u1-u0))
                vec_y_ = normalized(self.center_line.vec_y_spline(u))
                vec_y = normalized(vec_y_-projection(vec_y_, vec_z))

                if np.dot(vec_y, vec_y_) < 0:
                    vec_y *= -1

                vec_x = normalized(np.cross(vec_y, vec_z))
                
#                 rel_coord_in = (j*rel_coord_in1+(upsample_ratio-j)*rel_coord_in0)/upsample_ratio
                rel_coord_in = np.array([spl.spline(u) for spl in self.inner_wall_rel_spl_lon])[:,1:3]
                abs_coord_in = coord_rel_to_abs(rel_coord_in, cp, vec_y, vec_x, scale=0.5)
                
#                 rel_coord_ou = (j*rel_coord_ou1+(upsample_ratio-j)*rel_coord_ou0)/upsample_ratio
                rel_coord_ou = np.array([spl.spline(u) for spl in self.outer_wall_rel_spl_lon])[:,1:3]
                abs_coord_ou = coord_rel_to_abs(rel_coord_ou, cp, vec_y, vec_x, scale=0.5)
                
                
                
                yield vessel_p(cp, vec_z, vec_y, vec_x, BSpline3D(abs_coord_in, simple_init=True, periodic=True), 
                              BSpline3D(abs_coord_ou, simple_init=True, periodic=True))
        else:
            abs_coord_in1 = coord_rel_to_abs(rel_coord_in1, center1, vec_y1, vec_x1, scale=0.5)
            abs_coord_ou1 = coord_rel_to_abs(rel_coord_ou1, center1, vec_y1, vec_x1, scale=0.5)
            yield vessel_p(center1, vec_z1, vec_y1, vec_x1, BSpline3D(abs_coord_in1, simple_init=True, periodic=True), 
                              BSpline3D(abs_coord_ou1, simple_init=True, periodic=True))


#     def section(self, u):
#         if u == 1 or u == 0:
#             center, vec_z, vec_y = self.center_points[-1 if u == 1 else 0]
#             section_points = np.array(
#                 [spl.spline(u) for spl in self.longitudinal_spl])
#             return vessel_p(center, normalized(vec_z), normalized(vec_y), normalized(np.cross(vec_y, vec_z)), section_points)

#         b = np.digitize(u, self.center_line.u) - 1

#         u0, u1 = self.center_line.u[b:(b+2)]
#         if u0 == u:
#             center, vec_z, vec_y = self.center_points[b]
#             section_points = np.array(
#                 [spl.spline(u) for spl in self.longitudinal_spl])
#             return vessel_p(center, normalized(vec_z), normalized(vec_y), normalized(np.cross(vec_y, vec_z)), section_points)

#         cp = self.center_line.spline(u)

#         vec_z0, vec_z1 = self.center_points[b:(b+2), 1]
#         vec_z0 = normalized(vec_z0)
#         vec_z1 = normalized(vec_z1)

#         vec_y0, vec_y1 = self.center_points[b:(b+2), 2]
#         vec_y0 = normalized(vec_z0)
#         vec_y1 = normalized(vec_z1)

#         vec_z = ((u1-u)*vec_z0+(u-u0)*vec_z1)/(u1-u0)

#         vec_y = self.center_line.vec_y_spline(u) - cp
#         vec_y = normalized(vec_y-projection(vec_y, vec_z))

#         if np.dot(vec_y, ((u1-u)*vec_y0+(u-u0)*vec_y1)/(u1-u0)) < 0:
#             vec_y *= -1

#         vec_x = normalized(np.cross(vec_y, vec_z))

#         c = np.dot(vec_z, cp)

#         section_points = np.array(
#             [spl.spline(fsolve(lambda u, spl, vec_z, c:
#                                np.dot(spl(u), vec_z)-c, u,
#                                (spl.spline, vec_z, c), xtol=1e-9)[0])
#              for spl in self.longitudinal_spl])

#         return vessel_p(cp, vec_z, vec_y, vec_x, section_points)

    def guess_orifice(self):
        oct_distance = [np.mean(list(map(np.linalg.norm, oct_p-cen_p))) for (oct_p, cen_p)
                        in zip(self.inner_wall_abs_coords, self.center_points[:, 0])]
#         print(np.mean(oct_distance))
        us = UnivariateSpline(range(50, len(oct_distance)),
                              oct_distance[50:], s=len(oct_distance)/10)
        for i, v in enumerate(oct_distance):
            if v - us(i) < 0.4:
                break
        i = max(0, i-2)
        self.inner_wall_abs_coords = self.inner_wall_abs_coords[i:]
        self.inner_wall_rel_coords = self.inner_wall_rel_coords[i:]
        self.center_points = self.center_points[i:, ...]

    def plot(self, s=0, e=1.0, step=1, wall_points=None):
        if s is not None:
            s = int(self.center_points.shape[0] * s)

        if e is not None:
            e = int(self.center_points.shape[0] * e)

        fig = plt.figure(figsize=(8, 8))
        ax = Axes3D(fig)
        for p in self.center_points[s:e:step, 0, :]:
            ax.plot(*(p.reshape(-1, 3).T), 'ok', alpha=0.7)
        if wall_points:
            linspace = np.linspace(0, 1, wall_points+1)[:-1]
            for spl in self.section_spl[s:e:step]:
                ax.plot(*(spl.spline(linspace)).T, '.r', alpha=0.2)
        for ps in self.inner_wall_abs_coords[s:e:step]:
            ax.plot(*np.array(ps).T, 'ob', alpha=0.2)
        plt.show()


# In[ ]:


class BSpline3D:
    def __init__(self,
                 points,
                 group_average=False,
                 preprocess_linspace=False,
                 running_elements=False,
                 periodic=False,
                 simple_init=False):
        '''
        self.u: parametric 0.0 to 1.0 spline interpolation
        self.u2: 
        self.total_len: length of whole curve
        self.seg_len: length of each segment (between neighbor points)
        '''
        #         if group_average:
        #             self.spline, self.tck, self.u = self.init_spline(groupedAvg(points[:,0:3], group_average))
        #             self.vector1_spline,_,_ = self.init_spline(groupedAvg(points[:,0:3]+points[:,3:6], group_average))
        #         elif running_elements:
        #             self.spline, self.tck, self.u = self.init_spline(running_mean(points[:,0:3], running_elements))
        #             self.vector1_spline,_,_ = self.init_spline(running_mean(points[:,0:3]+points[:,3:6], running_elements))
        #         else:
        #         if periodic:
        #         self.spline, self.tck, self.u = self.init_spline(np.insert(points[:,0:3], -1, points[0,0:3], axis=0), per=periodic)
        #         if points.shape[1]>3:
        #             self.vector1_spline,_,_ = self.init_spline(np.insert(points[:,0:3], -1, points[0,0:3], axis=0)
        #                                                    +np.insert(points[:,3:6], -1, points[0,3:6], axis=0), per=periodic)
        #         else:
        self.spline, self.tck, self.u = self.init_spline(
            points[:, 0:3], per=periodic)
        if points.shape[1] > 3:
            self.vec_z_spline, _, _ = self.init_spline(
                points[:, 3:6], per=periodic)
        if points.shape[1] > 6:
            self.vec_y_spline, _, _ = self.init_spline(
                points[:, 6:9], per=periodic)
        
        
        self.points = points

        #         self.between_angles = []
        #         self.between_dist = []
        #         for (p1,p2),(_,p3) in zip(pairwise(points[0:-1,0:3]), pairwise(points[1:,0:3])):
        #             self.between_angles.append(np.rad2deg(angle_between(p2-p1, p3-p2)))
        #             self.between_dist.append(np.linalg.norm(p2-p1))
        #         self.between_angles=np.array(self.between_angles)
        #         self.between_dist=np.array(self.between_dist)

        self.periodic = periodic
        self.total_len = None
        self.drv1 = self.spline.derivative(nu=1)
        self.drv2 = self.spline.derivative(nu=2)
        if not simple_init:
            self.init2()
            if preprocess_linspace:
                samples = 1001 if preprocess_linspace is True else preprocess_linspace + 1
                self.ls = self.linspace(0.0, 1.0, samples)
                self.ls2 = np.linspace(0.0, 1.0, samples)
                self.ls_drv = np.array([self.drv1(t) for t in self.ls])
    
    
    def init2(self):
        # https://stackoverflow.com/questions/13384859/coefficients-of-spline-interpolation-in-scipy
        t,c,k = self.tck
        self.ppoly = np.swapaxes(np.array([PPoly.from_spline((t,c_,k), extrapolate=('periodic' if self.periodic else True)).c.T for c_ in c]),0,1)
        
        self.seg_len = []

        self.total_len = 0
        self.u2 = []
        self.u2.append(0.0)
        for u1, u2 in pairwise(self.u):
            l = self.est_length(u1, u2, sample_points=100, precision=1e-7)
            self.total_len += l
            self.seg_len.append(l)
            self.u2.append(self.total_len)

        self.seg_len = np.array(self.seg_len)
        self.u2 = np.array(self.u2)
        self.u2 /= self.total_len

        self.ls = None

    def init_spline(self, data, s=0, k=3, per=False):
        if per:
            tck, u = interpolate.splprep(
                np.append(data, [data[0]], axis=0).T, s=s, nest=-1, k=k, per=per, 
                u=np.linspace(0,1,data.shape[0]+1))
#             tck, u = interpolate.splprep(data.T, s=s, nest=-1, k=k, per=per, 
#                 u=np.linspace(0,1,data.shape[0]), quiet=1)
        else:
            tck, u = interpolate.splprep(data.T, s=s, nest=-1, k=k, per=per, quiet=1, u = np.linspace(0,1,data.shape[0]))
        
        spline = interpolate.BSpline(
            np.array(tck[0]),
            np.array(tck[1]).T, tck[2], extrapolate = 'periodic' if per else True)
        return spline, tck, u

#     def distances_var(self, t, a, b):
#         atb = np.concatenate(([a], t, [b]))   # added endpoints
#         y = self.spline(atb)
#         dist_squared = np.diff(atb)**2 + np.diff(y)**2
#         return np.var(dist_squared)

    def linspace(self, s=0.0, e=1.0, n=1000):
        ls = np.linspace(s, e, n)
        #         ls2 = np.linspace(s,e,10*n)
        #         spl = self.spline(ls2)
        #         pt = interparc(ls, spl[:,0], spl[:,1], spl[:,2])
        #         return pt
        # ret = np.zeros_like(ls)
        # for i in range(ls.shape[0]):
        #     ret[i] = self.u2_u(ls[i])
        ret = np.array([self.u2_u(i, precision=(e - s) / n / 10) for i in ls])
        return ret

    def linvalue(self, s=0.0, e=1.0, n=1000, return_u=False):
        ret_u = self.linspace(s, e, n)
        if return_u:
            return self.spline(ret_u), ret_u
        else:
            return self.spline(ret_u)
   
    
    def u2_u(self, t, precision=1e-5, maxLoop=100, recursive=True):
        '''
        tranform u2 to u
        '''
        if t == 0.0 or t == 1.0:
            return t

        if self.total_len is None:
            self.init2()

        if self.ls is None:
            #         if True:
            u_cu = self.u2
            u_sc = self.u
        else:
            u_cu = self.ls2
            u_sc = self.ls

        b = np.digitize(t, u_cu) - 1
        #         print(str(b)+','+str(t))
        #         print(u_sc[b])
        #         print(u_cu[b])
        ret = np.interp(t, u_cu[b:(b + 2)], u_sc[b:(b + 2)])
        if not recursive:
            return ret
        u = est_l = u_sc[b]
        est_u = u_sc[b + 1]
        cum_l = l = u_cu[b]
        cum_u = u_cu[b + 1]

        prv_d, prv_ret = 9999, ret
        d = 1e-3
        loop_i = 0
        while 1:
            loop_i += 1
            #             if loop_i>maxLoop:
            #                 break
            d = (self.est_length(u, ret, precision) / self.total_len + l) - t
            #             if abs(d)>abs(prv_d):
            #                 ret = prv_ret
            #                 break
            if d == 0 or (abs(d) < precision) or est_u == est_l:
                break
            prv_d, prv_ret = d, ret
            if d > 0:
                est_u = ret
                cum_u = d + t
            else:
                est_l = ret
                cum_l = d + t
            ret = np.interp(t, [cum_l, cum_u], [est_l, est_u])

#         print(loop_i)
        return ret

    def intersect_u(self, start_p, vector, return_coord=False, threshold=1e-9, max_iter=1000):
        '''
        for closed B-spline, find the intersection with the vector start from start_p, return coresponding u or coord
        '''
        u1, u2 = 0.0, 1.0
        v1 = normalized(self.spline(u1)-start_p)
        v2 = normalized(self.spline(u2)-start_p)

        target = normalized(vector)
        ret = None
        
        i = 0
        while 1:
            i+=1
            
            if i>max_iter:
                raise Exception('max iterations reached!')
            a1 = np.dot(v1, target)
            a2 = np.dot(v2, target)
            if 1-a1 < threshold:
                ret = u1
                break
            elif 1-a2 < threshold:
                ret = u2
                break
            

            u_ = (u1+u2)/2
            v_ = normalized(self.spline(u_)-start_p)

            i1 = np.dot(target, self.spline((u_+u1)/2)-start_p)
            i2 = np.dot(target, self.spline((u_+u2)/2)-start_p)

            if abs(i1-i2) < threshold:
                ret = u_
                break
            elif i1 > i2:
                u2 = u_
                v2 = v_
            elif i2 > i1:
                u1 = u_
                v1 = v_

        return ret if not return_coord else self.spline(ret)
    
    def intersect_plane(self, start_u_range, center_point, normal_vec_of_plane, return_coord=False, threshold = 1e-9, max_iter=1000):
        u0,u1 = min(start_u_range), max(start_u_range)
        
        vec_z = normalized(normal_vec_of_plane)
        cp = center_point
        spl = self.spline
        i=0
        ret = None
        
        while 1:
            i+=1
            if i>max_iter:
                print(start_u_range, u0, u1)
                raise Exception('max iterations reached!')
            dot0 = np.dot(spl(u0)-cp, vec_z)
            dot1 = np.dot(spl(u1)-cp, vec_z)
            if dot0 * dot1 <=0:
                break
            
            if dot1 > dot0:
                u0, u1 = u1, u1+(u1-u0)
            else:
                u0, u1 = u0 - (u1-u0), u0
        
        i=0
        while 1:
            i+=1
            
            if i>max_iter:
                print(start_u_range, u0, u1)
                raise Exception('max iterations reached!')
            
            dot0 = abs(np.dot(spl(u0)-cp , vec_z))
            dot1 = abs(np.dot(spl(u1)-cp , vec_z))
            if dot0<threshold:
                ret = u0
                break
            elif dot1<threshold:
                ret = u1
                break
            
            
            u = (u0+u1)/2
            vec = normalized(spl(u) - cp)
            if abs(np.dot(vec, vec_z)) < threshold:
                ret= u
                break
            if dot0 < dot1:
                u1 = u
            else:
                u0 = u
                
        return ret if not return_coord else spl(ret)
            
            
        

    def est_length(self, s=0.0, e=1.0, sample_points=10, precision=1e-5):
        if s == e:
            return 0.0
        n = np.count_nonzero(np.logical_and(
            self.u >= s, self.u <= e))  # how many real data points
        n = n * 10 + 1 if n > 0 else 10
        ls = np.linspace(s, e, n)
        v = self.spline(ls)
        sp = (e - s) / (n - 1) / 2
        center_points = self.spline(np.linspace(s + sp, e - sp, n - 1))
        l = 0
        for (a, b), c, (i1, i2) in zip(
                pairwise(v), center_points, pairwise(ls)):
            seg_len = np.linalg.norm(b - a)
            triang_len = np.linalg.norm(c - a) + np.linalg.norm(c - b)
            
            if triang_len < precision:
                l += triang_len
            else:
                delta_len = (triang_len - seg_len) / triang_len
                l += triang_len if delta_len < precision else self.est_length(
                    i1, i2, 10, precision)
        return l

    def drv(self, t, nu=1):
        u0 = t
        u1 = u0 + 1e-5
        s0 = self.spline(u0)
        s1 = self.spline(u1)
        v = s1 - s0
        if nu == 1:
            ret = self.drv1(u0)
        elif nu == 2:
            ret = self.drv2(u0)
        else:
            ret = self.spline.derivative(nu)(u0)

        prod = np.dot(v, ret)
        #         print(prod)
        return ret if prod > 0 else ret * (-1)

    def smooth_derivative(self, t, nearest=0.01, samples=21):
        if t < nearest:
            rng = [0, nearest * 2]
        elif t > 1 - nearest:
            rng = [1 - nearest * 2, 1]
        else:
            rng = [t - nearest, t + nearest]

        ret = []
        for t in np.linspace(rng[0], rng[1], samples):
            ret.append(normalized(self.ls_drv[int(np.floor(t * 10))]))
        return np.mean(np.array(ret), axis=0)

    def derivative(self, t):
        return self.drv(self.u2_u(t), nu=1)

    def derivative2(self, t):
        return self.drv(self.u2_u(t), nu=2)

    def plot(self, s=0.0, e=1.0, n=100, show_tangent=False):

        if show_tangent:
            ls = np.linspace(s, e, n)
            u = np.array([self.u2_u(i) for i in ls])
            p = self.spline(u)
            seg_len = (e - s) / n * 0.7
            tan = np.array([normalized(self.drv(i)) for i in u]) * seg_len
            tan_lines_end = p + tan
            lines = [np.array([p1, p2]) for p1, p2 in zip(p, tan_lines_end)]
            lc = Line3DCollection(np.array(lines))
        else:
            p = self.linvalue(s, e, n)


#         print(p)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_aspect('equal')
        ax.plot(*([p[:, i] for i in range(3)]), '-o', markersize=3)
        if show_tangent:
            ax.add_collection3d(lc)
        ax.set_xlabel('Z')
        ax.set_ylabel('Y')
        ax.set_zlabel('X')
        set_axes_equal(ax)
        plt.show()


def test_cubic_spline():
    centerlines = get_centerlines(
        r'C:\Users\Administrator\Downloads\CCTA\CCA results 75% 11 TI - 1109')
    points = centerlines['LAD']

    bsp = BSpline3D(points)

    # 3D example
    total_rad = 10
    z_factor = 3
    noise = 0.1

    num_true_pts = 10
    s_true = np.linspace(0, total_rad, num_true_pts)
    x_true = np.cos(s_true)
    y_true = np.sin(s_true)
    z_true = s_true / z_factor

    num_sample_pts = 80
    s_sample = np.linspace(0, total_rad, num_sample_pts)
    x_sample = np.cos(s_sample) + noise * np.random.randn(num_sample_pts)
    y_sample = np.sin(s_sample) + noise * np.random.randn(num_sample_pts)
    z_sample = s_sample / z_factor + noise * np.random.randn(num_sample_pts)

    tck, u = interpolate.splprep([points[:6, i] for i in range(3)],
                                 s=0,
                                 nest=-1)
    sp = interpolate.BSpline(np.array(tck[0]), np.array(tck[1]).T, tck[2])
    x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
    u_fine = np.linspace(0, 1, num_true_pts)
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)

    # print([np.array(interpolate.splev(u[i], tck)).reshape(3) - points[i, :3] for i in range(6)])

    fig2 = plt.figure(2)
    ax3d = fig2.add_subplot(111, projection='3d')
    # ax3d.plot(x_true, y_true, z_true, 'b')
    # ax3d.plot(x_sample, y_sample, z_sample, 'r*')
    # ax3d.plot(x_knots, y_knots, z_knots, 'go')
    ax3d.plot(*([points[:6, i] for i in range(3)]), 'go')
    ax3d.plot(x_fine, y_fine, z_fine, 'g')
    fig2.show()
    plt.show()


# In[ ]:


def mask_overlay_data(data, mask, mask2=None, window=(500, 2000), figsize=(8, 8), output_dir=None):
    new_data = np.repeat(data[..., np.newaxis], 3, axis=-1).astype(np.float64)
    
    new_mask = np.zeros(mask.shape)
    new_mask = np.repeat(new_mask[..., np.newaxis], 3, axis=-1)
    new_mask[..., 0] = mask
    
    new_mask2 = np.zeros(mask.shape)
    new_mask2 = np.repeat(new_mask2[..., np.newaxis], 3, axis=-1)
        
    if mask2 is not None:
        new_mask2[..., 1] = mask2
    
    w1, w2 = window[0]-window[1]/2, window[0] + window[1]/2
    new_data = (np.clip(new_data, w1, w2) - w1)/(w2-w1)
    
    if output_dir:
        d = Path(output_dir)
        if not d.exists():
            os.makedirs(str(d.resolve()))
        for i, sl in enumerate(new_data+new_mask+new_mask2):
            plt.imsave(str(d.joinpath('{}.png'.format(i))), sl)
    else:
        ViewCT(new_data+new_mask+new_mask2, window=(0.5, 0.5),
               cmap=None, subplots=(1, 1), interpolation=None, figsize=figsize)


# In[ ]:


def plots(data, **kwargs):
    fig, ax = plt.subplots(data.shape[0], 1, figsize=(8, 5*data.shape[0]))

    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'gray'

    for x, pl in zip(ax.ravel(), data):
        x.imshow(pl, **kwargs)

    plt.tight_layout()


# In[2]:


def straighten_data_mask(ctVolume, cor, precision=(1, 1), show_range=(0.0, 1.0), output_dim=(100, 100), output_spacing=None, upsample_z=1, average_z=False, no_fill=False, outer_wall=False):
    def set_mask(point_coord):
        vec = point_coord - cp
        y = np.dot(vec, y_axis)
        x = np.dot(vec, x_axis)
        idx = np.array([y, x]) / output_spacing * precision + center_ind
        idx = np.clip(idx, -0.5, new_dim-0.500001)
        idx = np.insert(idx, 0, z, axis=0)
#         sp.append(idx)
        all_coord.append(idx)

    precision = nparr(precision, 2)
    output_dim = nparr(output_dim, 2)
    new_dim = output_dim * precision
    

    _t = time()

    cmpr, pixel_spacing, vessel_ps = curved_MPR(
        ctVolume, cor, output_dim=output_dim, output_spacing=output_spacing, upsample_z=upsample_z)
#     ret = np.empty((cor.center_points.shape[0], new_dim[0], new_dim[1]))

    if output_spacing is None:
        output_spacing = pixel_spacing

    min_sp = output_spacing / np.amax(precision)

    center_ind = (new_dim) / 2.0
    all_coord = []
#     sp = []

#     for i, rel_coords in cor.inner_wall_rel_coords.items():
#         plane = np.zeros(new_dim)
#         int_coords = np.floor(rel_coords + center_ind +0.5).astype(np.int)
#         plane[tuple(int_coords.T)]=1
#         ret[i,...]=plane

    # abs coords conversion
    
    
    results = {}
    not_filled=[]
    results2 = {}
    not_filled2=[]
    point_count = cor.center_points.shape[0]
    ret = np.zeros((point_count + (point_count-1)*(upsample_z-1), new_dim[0], new_dim[1]))
    
    print('straighten: {}'.format(time()-_t))
    _t = time()
    
    
#     for z, rel_coords in enumerate(cor.inner_wall_rel_coords):
#         spl = CubicSpline(np.linspace(0,1,rel_coords.shape[0]+1), 
#                           np.append(rel_coords, [rel_coords[0]], axis=0), 
#                           bc_type='periodic')
        
#         n2 = np.ceil(
#             3 * 1000 / min_sp *
#             np.linalg.norm(spl(0) - spl(1 / 1000))
#         ).astype(np.int)
        
#         rel_coords = spl(np.linspace(0,1,n2))

#         rel_coords /= min_sp * 2
        
#         rel_coords += center_ind
#         rel_coords = np.floor(rel_coords+0.5).astype(np.int)
#         rel_coords = np.clip(rel_coords, 0, np.array(ret.shape[1:3])-1)
        
#         rel_coords = np.append(rel_coords, [np.floor(center_ind+0.5).astype(np.int)], axis=0)
#         rel_coords = np.insert(rel_coords, 0, z, axis=1)
#         ret[tuple(rel_coords.T)] = 1
        
#     for z, rel_coords in enumerate(cor.outer_wall_rel_coords):
#         spl = CubicSpline(np.linspace(0,1,rel_coords.shape[0]+1), 
#                           np.append(rel_coords, [rel_coords[0]], axis=0), 
#                           bc_type='periodic')
        
#         n2 = np.ceil(
#             3 * 1000 / min_sp *
#             np.linalg.norm(spl(0) - spl(1 / 1000))
#         ).astype(np.int)
        
#         rel_coords = spl(np.linspace(0,1,n2))

#         rel_coords /= min_sp * 2
        
#         rel_coords += center_ind
#         rel_coords = np.floor(rel_coords+0.5).astype(np.int)
#         rel_coords = np.clip(rel_coords, 0, np.array(ret.shape[1:3])-1)
        
#         rel_coords = np.append(rel_coords, [np.floor(center_ind+0.5).astype(np.int)], axis=0)
#         rel_coords = np.insert(rel_coords, 0, z, axis=1)
#         ret2[tuple(rel_coords.T)] = 1

#     return ret, ret2, cmpr
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=30) as executor:
        #         for z, (center_point, spl) in enumerate(zip(cor.center_points, cor.section_spl)):
        for z, vessel_p in enumerate(vessel_ps):
#         for z, vessel_p in enumerate(vessel_ps[918:920]):
            #         plane = np.empty(new_dim)

            #         u0 = spl.intersect_u(cp, center_point[2])

            results[executor.submit(straighten_sub, vessel_p, vessel_p.inner_wall, precision,
                                    output_spacing, center_ind, new_dim, z, min_sp, no_fill)] = z
             
#             results[z] = straighten_sub(vessel_p, vessel_p.inner_wall, precision, output_spacing, center_ind, new_dim, z, min_sp, no_fill)
#             break

        for fs in concurrent.futures.as_completed(results):
            z = results[fs]
            res, is_filled = fs.result()
            ret[z, ...] = res
            if not is_filled:
                not_filled.append(z)
        
        if outer_wall:
            ret2 = np.zeros((point_count + (point_count-1)*(upsample_z-1), new_dim[0], new_dim[1]))

            for z, vessel_p in enumerate(vessel_ps):  
                results2[executor.submit(straighten_sub, vessel_p, vessel_p.outer_wall, precision,
                                        output_spacing, center_ind, new_dim, z, min_sp, no_fill)] = z
            for fs in concurrent.futures.as_completed(results2):
                z = results2[fs]
                res, is_filled = fs.result()
                ret2[z, ...] = res
                if not is_filled:
                    not_filled2.append(z)
        
        if not no_fill:
            filling_from_corner(ret, not_filled)
            if outer_wall:
                filling_from_corner(ret2, not_filled2)
        print(time()-_t)
        _t = time()
    #     ret = output_mask((cor.center_points.shape[0], output_dim[0], output_dim[1]), np.insert(
    #         precision, 0, 1, axis=0), result_coord, _t)
        if np.amax(precision) > 1:
            ret = groupedAvg(ret, precision, axis=(1, 2))
            if outer_wall:
                ret2 = groupedAvg(ret2, precision, axis=(1,2))
        
        if average_z and upsample_z>1:
            ret = np.vstack(([ret[0]], groupedAvg(ret[1:], (upsample_z, ), axis=(0, ))))
            cmpr = np.vstack(([cmpr[0]], groupedAvg(cmpr[1:], (upsample_z, ), axis=(0, ))))
            
            if outer_wall:
                ret2 = np.vstack(([ret2[0]], groupedAvg(ret2[1:], (upsample_z, ), axis=(0, ))))
        
#     for coor in result_coord:
#         coor = np.floor(np.array(coor) + 0.5).astype(np.int)
#         ret[tuple(coor.T)] = 1
#     all_coord = np.floor(np.array(all_coord)+0.5).astype(np.int)
#         plane[tuple(all_coord.T)]=1
#     ret[tuple(all_coord.T)] = 1

#     for i, prec in enumerate(precision):
#         if prec != 1:
#             ret = groupedAvg(ret, prec, axis=i+1)
    
    if outer_wall:
        return ret, ret2, cmpr
    else:
        return ret, cmpr

def consecutive_number_ranges(nums):
    ### https://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))
    
def filling_from_corner(mask, z):
    if not z:
        return 
    
    for s, e in consecutive_number_ranges(z):
        votes=np.zeros((2,2),dtype=np.int)
        for i in (s-1, e+1):
            try:
                plane = mask[i]
                for j, y in enumerate((0, plane.shape[0]-1)):
                    for k, x in enumerate((0, plane.shape[1]-1)):
                        votes[j,k]+=(1-plane[y,x])
            except:
                pass
        
        ## fill background
        max_v = np.amax(votes)
        i, j = np.where(votes==max_v)
        seed_options = ((0, plane.shape[0]-1),(0, plane.shape[1]-1))
        for k in range(s,e+1):
            bg = mask[k].copy()
            bg = np.ascontiguousarray(bg, dtype=np.uint8)
            for seed_i, seed_j in zip(i,j):
                seed = tuple((seed_options[0][seed_i], seed_options[1][seed_j]))[::-1]
                try:
                    cv2.floodFill(bg, None, tuple(seed), 1, 0, 0, cv2.FLOODFILL_FIXED_RANGE)
                except:
                    print('Flood fill failed at plane #{}'.format(k))
                    continue
            mask[k] += (1-bg)


def straighten_sub(vessel_p, spl, precision, output_spacing, center_ind, new_dim, z, min_sp, no_fill):

    all_coord = []

    cp = vessel_p.center
#     spl = BSpline3D(vessel_p.wall_points, periodic=True, simple_init=True)
    
    n2 = np.ceil(
        3 * 1000 / min_sp *
        np.linalg.norm(spl.spline(0) - spl.spline(1 / 1000))
    ).astype(np.int)
#         n2_linspace = np.mod(np.linspace(u0, u0 + 1 - 1 / n2, n2), 1.0)
    n2 = max(n2, 2)
    n2_linspace = np.linspace(0, 1.2, 4*n2)
    n3 = np.ceil(n2 / np.pi / 2).astype(np.int)
    n3 = max(n3, 1)
    z_axis = vessel_p.vec_z
    y_axis = vessel_p.vec_y
    x_axis = vessel_p.vec_x

    axes = np.array([y_axis, x_axis]).T

    section_points = spl.spline(n2_linspace)
#     set_mask(cp)
#     straighten_set(cp, all_coord, cp, precision, output_spacing,
#                    center_ind, axes, new_dim, z)
#     plot3d(vessel_p.wall_points)
    rel_coord = np.matmul(section_points-cp, axes) / min_sp + center_ind
#     rel_coord_copy = rel_coord.copy()
    
    rel_coord = np.floor(rel_coord+0.5).astype(np.int)

    min_x, max_x = minmax(rel_coord[:, 1])
    min_y, max_y = minmax(rel_coord[:, 0])

    origin = np.array([min_y, min_x])
    
#     print(rel_coord, origin)
    
    rel_coord -= origin
#     rel_coord_copy-=origin
    center_ind-=origin

    shape = (max_y-min_y+1, max_x-min_x+1)
    
#     arr = np.zeros(shape, dtype=np.int)
#     arr[tuple(rel_coord.T)]=1
    arr = coo_matrix((np.repeat(
        1, rel_coord.shape[0]), rel_coord.T), shape=shape, dtype=np.int).toarray()
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    seed_on_boundary = False
    cv_floodfill_error=False
    if not no_fill:
        

    #     print(rel_coord)
        seed = np.floor(
                center_ind+0.5).astype(np.int)[::-1]
        seed_on_boundary = False
        for p in rel_coord:
            if np.all(seed==p):
                seed_on_boundary=True
                break
                
        
        
        try:
            cv2.floodFill(arr, None, tuple(seed), 1, 0, 0, cv2.FLOODFILL_FIXED_RANGE)
        except:
            cv_floodfill_error=True
#             pass
#             closest_ds = np.amin(np.abs(rel_coord_copy - center_ind+origin))
#             except:
#                 print(np.floor(
#                     center_ind-origin+0.5).astype(np.int), arr.shape)
#             found=False
#             for i, point0 in enumerate(rel_coord_copy[:-1]):
#                 for p in rel_coord_copy[i+1:]:
#                     try:
#                         cv2.floodFill(arr, None, tuple(np.floor(
#                             (point0+p)/2+0.5).astype(np.int)), 1, 0, 0, cv2.FLOODFILL_FIXED_RANGE)
#                         found=True
#                     except:
#                         pass
#                 if found:
#                     break
#             del arr
#             del rel_coord
#             del rel_coord_copy
#             del section_points
#             ratio = np.ceil(1/closest_ds)
#             ret = straighten_sub(vessel_p, spl, precision, output_spacing/ratio, center_ind*ratio, new_dim*ratio, z, min_sp/ratio, no_fill)
            
#             return groupedAvg(ret, (2,2), axis=(0,1))

    ret = np.zeros(new_dim)
    ret_coord = np.array(np.where(arr > 0)).T + origin
    
    ## some indexes are out of boundary of array
#     ret_coord = np.clip(ret_coord, 0, new_dim-1)
    ret_coord_=[]
    for coord in ret_coord:
        if (coord>=0).all() and (coord<new_dim).all():
            ret_coord_.append(coord)
    ret_coord = np.array(ret_coord_)
    
    ret[tuple(ret_coord.T)] = 1
    
#     if seed_on_boundary:
#         print(z)
#     plt.imshow(ret, cmap = 'gray')
    return (ret, not (cv_floodfill_error or seed_on_boundary))

    for p in section_points:
        #             sp.append(p)
        for j in range(n3):
            straighten_set(((n3 - j) * p + j * cp) / n3, all_coord, cp, precision,
                           output_spacing, center_ind, axes, new_dim, z)
#             set_mask(((n3 - j) * p + j * cp) / n3)
#                 break

    return all_coord


def straighten_set(point_coord, all_coord, cp, precision, output_spacing, center_ind, axes, new_dim, z):
    idx = np.matmul(point_coord - cp, axes) /         output_spacing * precision + center_ind
    idx = np.clip(idx, -0.5, new_dim-0.500001)
    idx = np.insert(idx, 0, z, axis=0)
#         sp.append(idx)
    all_coord.append(idx)


# In[ ]:


# %%pixie_debugger


def plot_sec(ct, cor, n):
    spl = cor.section_spl[n]
    cp = cor.center_points[n, 0]

    n2 = 30
    sp = spl.spline(np.linspace(0, 1-1/n2, n2))
    mp = []
    clr = []
    n3 = int(n2/2)
    for p in sp:
        clr.append(
            ct.data[tuple(np.floor(ct.absolute_to_pixel_coord(p, True)).astype(np.int).T)])
        for i in range(n3):
            p_ = ((n3-i)*p + i*cp)/n3
            mp.append(p_)
            clr.append(
                ct.data[tuple(np.floor(ct.absolute_to_pixel_coord(p_, True)).astype(np.int).T)])

    sp = np.concatenate((sp, np.array(mp)), axis=0)
    clr = np.array(clr).astype(np.float64)
    clr -= np.amin(clr)
    clr /= np.amax(clr)
    clr = np.repeat(clr[..., np.newaxis], 3, axis=-1)

    fig = plt.figure(figsize=(10, 10))
    ax = Axes3D(fig)
    ax.scatter(*sp.T, c=clr)
    set_axes_equal(ax)
    plt.show()

# plot_sec(ct, cor, 30)


# In[ ]:





# In[ ]:


def generate_mask(coronary, ctVolume, precision=(1, 1, 1), out=None, no_fill=False):
    def set_mask(point_coor):
        #         all_points.append(point_coor)
        #         return
        float_ind = ctVolume.absolute_to_pixel_coord(
            point_coor, return_float=True)
        float_ind = float_ind * precision

#         float_ind = np.floor(float_ind + 0.5).astype(np.int)
        float_ind = np.clip(float_ind, -0.5, shape1-0.500001)
        all_coord.append(float_ind)
#         all_coord.add(np.dot(float_ind, base))
#         try:
#             mask[tuple(float_ind)] = 1
#         except:
#             print('Coordinate {}, ind {}'.format(
#                 point_coor,
#                 np.abs(
#                     ctVolume.absolute_to_pixel_coord(
#                         point_coor, return_float=True))))

    data, origin, spacing = ctVolume.data, ctVolume.origin, ctVolume.spacing
    shape0 = np.array(data.shape)
    shape1 = shape0 * precision
    base = np.array([shape1[1]*shape1[2], shape1[2], 1]).astype(np.int)
    precision = nparr(precision)
#     ret = np.zeros(np.array(data.shape) * precision, dtype=np.int)
    new_spacing = np.abs(spacing / precision)
    min_sp = np.amin(new_spacing)
    #         center_points, spl_u = coronary.center_line.linvalue(n=20, return_u=True)
    #     spl_u = np.linspace(0,1,u_samples+1)
    #     center_points = coronary.center_line.spline(spl_u)

    #     for center, u in zip(coronary.center_points, spl_u):
    #         section_points = coronary.BSplineSurfacePoint(u)

    #         for p in section_points:
    #             set_mask(p, ret, ctVolume, precision)

    #             v = (p-center)/radial_samples
    #             for i in range(radial_samples):
    #                 set_mask(center+v*i, ret, ctVolume, precision)
    #     center_points_diff = np.diff(coronary.center_points)

#     all_coord = set()
    all_coord = []
    _t = time()
    ctInfo = dict(spacing=spacing, origin=origin)
#     if hasattr(coronary, 'longitudinal_spl'):
    if 1:
        
        if no_fill:
            n1 = 1
        else:
            n1 = np.ceil(coronary.center_line.total_len / min_sp / coronary.center_line.u.shape[0]).astype(np.int)
            n1 = max(n1, 2)
        results = []
        result_coord = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=30) as executor:
#             for i, u in enumerate(np.linspace(0, 1, 4*n1)):
            for i, vessel_p in enumerate(coronary.upsample_sections(10)):
#                 center_vec = coronary.center_line.center_vec(u)
                
#                 section_points0 = np.array(coronary.BSplineSurfacePoint(u))
                results.append(executor.submit(generate_mask_sub3, ctInfo,
                                               vessel_p, precision, shape1, min_sp, i, no_fill))
                
#                 results.append(generate_mask_sub3(ctInfo,
#                                                center_vec, section_points0, precision, shape1, min_sp, i))
#                 break

            for fs in concurrent.futures.as_completed(results):
                result_coord.append(fs.result())

    else:
        n = int(coronary.center_points.shape[0]/10) * 2
        i1, i2 = 0, n
        results = []
        result_coord = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=30) as executor:
            for i in range(0, n, 2):
                results.append(executor.submit(generate_mask_sub, ctInfo,
                                               coronary.center_points[i:(
                                                   i+2), 0, :],
                                               coronary.section_spl[i:(i+2)],
                                               coronary.section_u0[i:(i+2)],
                                               precision, shape1, min_sp, i, True))
            results.append(executor.submit(generate_mask_sub, ctInfo,
                                           coronary.center_points[n:, 0, :],
                                           coronary.section_spl[n:],
                                           coronary.section_u0[n:],
                                           precision, shape1, min_sp, n))

            for fs in concurrent.futures.as_completed(results):
                result_coord.append(fs.result())
#         n=int(coronary.center_points.shape[0]/10)
#         for i, ((p0, p1), (spl0, spl1), (u0,u1)) in enumerate(zip(
#                 pairwise(coronary.center_points[:, 0, :]),
#                 pairwise(coronary.section_spl),
#                 pairwise(coronary.section_u0))):
#             n1 = np.ceil(3 * np.linalg.norm(p0 - p1) / min_sp).astype(np.int)
#             n1 = max(n1,2)
#             n2 = np.ceil(
#                 3 * 1000 /
#                 min_sp * np.linalg.norm(spl0.spline(0) - spl0.spline(1 / 1000))
#             ).astype(np.int)
#             if i<n:
#                 n1 *= 2
#                 n2 = max(n2*2,2)
#                 n3 = np.ceil(n2 / np.pi / 2).astype(np.int)
#                 n3 = max(n3,1)

#             else:
#                 n2 = max(n2,2)
#                 n3 = np.ceil(n2 / np.pi / 2).astype(np.int)
#                 n3 = np.clip(n3,1,5)
#             n2_linspace = np.linspace(0, 1 - 1 / n2, n2)
#             this_section_points = spl0.spline(np.mod(n2_linspace+u0,1.0))
#             next_section_points = spl1.spline(np.mod(n2_linspace+u1,1.0))
#             for i in range(n1):
#                 section_points = (
#                     (n1 - i) * this_section_points + i * next_section_points) / n1
#                 center_point = ((n1 - i) * p0 + i * p1) / n1
#                 set_mask(center_point)
#                 for p in section_points:
#                     for j in range(n3):
#                         set_mask(((n3 - j) * p + j * center_point) / n3)

#         else:
#             set_mask(p1)
#             for p in next_section_points:
#                 for j in range(n3):
#                     set_mask(((n3 - j) * p + j * p1) / n3)

#     ret[tuple(np.array(all_coord).T)]=1
#     ret = ret.ravel()
#     ret[tuple(all_coord),]=1
#     ret = ret.reshape(shape1)
    print('generate mask: {}'.format(time()-_t))
    _t = time()


#     for coor in result_coord:
#         coor = np.floor(np.array(coor) + 0.5).astype(np.int)
#         ret[tuple(coor.T)] = 1
#     all_coord = np.floor(np.array(all_coord) + 0.5).astype(np.int)
#     ret[tuple(all_coord.T)] = 1


#     ret = sparse.COO(result_coord.T, data=1, shape=shape1)


#     props = regionprops(label(ret, background=-1, connectivity=2))
#     for prop in props[1:]:
#         if ret[tuple(prop.coords[0,:])]==1:
#             bbox = prop.bbox
#             ret[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]] = prop.filled_image.astype(ret.dtype)
#             break

#     ret = binary_fill_holes(ret)

#     fill_struct = np.zeros((3,3,3))
#     fill_struct[1, 1, 0:3] = fill_struct[1, 0:3, 1] = fill_struct[0:3,1,1] = 1
#     ret = binary_fill_holes(ret, structure = fill_struct)

#     for i, prec in enumerate(precision):
#         if prec != 1:
#             ret = groupedAvg(ret, prec, axis=i)
#     ret = convolve(ret, np.ones(precision)/np.prod(precision))

    ret = output_mask(shape0, precision, result_coord, _t)

    if out is None:
        return ret
    else:
        out = ret


def output_mask(original_shape, precision, result_coord, _t=None):

    if 0:
        shape1 = original_shape * precision
        base = np.array([shape1[1]*shape1[2], shape1[2], 1])
        result_coord = list(chain(*result_coord))
        result_coord = np.floor(np.array(result_coord) + 0.5).astype(np.int)

    #     result_coord = np.clip(result_coord, 0, shape1-1)
        result_coord = np.matmul(result_coord, base)
        result_coord = np.unique(result_coord, axis=0)

        zs = np.floor(result_coord / base[0]).astype(np.int)
        result_coord -= base[0] * zs
        ys = np.floor(result_coord / base[1]).astype(np.int)
        result_coord -= base[1] * ys
        xs = np.floor(result_coord).astype(np.int)
    else:
        shape1 = original_shape * precision
        ret = np.zeros(shape1)
        for coord in result_coord:
            coord = np.floor(coord + 0.5).astype(np.int)
            ret[tuple(coord.T)] = 1

#     print(zs.min(), zs.max())
#     print(ys.min(), ys.max())
#     print(xs.min(), xs.max())

    if _t:
        print(time()-_t)
        _t = time()

    if 0:
        ret = np.zeros(original_shape, dtype=np.float64)
        val = 1.0/np.prod(precision)
        for a, b, c in zip(zs, ys, xs):
            coord = np.floor(np.array([a, b, c]) / precision).astype(np.int)
            ret[tuple(coord)] += val
    else:
        ret = ret.reshape((original_shape[0], precision[0],
                           original_shape[1], precision[1],
                           original_shape[2], precision[2])).mean(axis=(1, 3, 5))

    ret = np.clip(ret, 0, 1)

    if _t:
        print(time()-_t)
    return ret


# In[ ]:


def generate_mask_set(point_coor, all_coord, ctInfo, precision, shape1):
    #         all_points.append(point_coor)
    #         return
    float_ind = (point_coor-ctInfo['origin'])/ctInfo['spacing']
    float_ind = float_ind * precision

#         float_ind = np.floor(float_ind + 0.5).astype(np.int)
    float_ind = np.clip(float_ind, -0.5, shape1-0.500001)
    all_coord.append(float_ind)


def generate_mask_sub3(ctInfo, vessel_p, precision, shape1, min_sp, i, no_fill):
    _t = time()
    all_coord = []
    
#     spl = BSpline3D(vessel_p.wall_points, simple_init=True)
    spl = vessel_p.inner_wall
    n2 = np.ceil(
        3 * 1000 /
        min_sp * np.linalg.norm(spl.spline(0) - spl.spline(1 / 1000))
    ).astype(np.int)
    
    
    n2_linspace = np.linspace(0, 1, n2 * 4)
    section_points = spl.spline(n2_linspace)
    point_coor = fill_lumen(section_points, vessel_p, min_sp)
    float_ind = (point_coor-ctInfo['origin'])/ctInfo['spacing']
    float_ind *= precision
    float_ind = np.clip(float_ind, -0.5, shape1-0.500001)
#     print(float_ind.shape)
#     print(float_ind)
#     plot3d(float_ind)
    
    return float_ind

    if i < 10:
        n2 = max(n2*2, 2)
        n3 = np.ceil(n2 / np.pi / 2).astype(np.int)
        n3 = max(n3, 1)

    else:
        n2 = max(n2, 2)
        n3 = np.ceil(n2 / np.pi / 2).astype(np.int)
        n3 = max(n3, 1)

    if no_fill:
        n3 = min(n3, max(precision))
        
    n2_linspace = np.linspace(0, 1 - 1 / n2, n2)

    section_points = spl.spline(n2_linspace)
    for p in section_points:
        for j in range(n3):
            #             set_mask(((n3 - j) * p + j * center_point) / n3)
            generate_mask_set(((n3 - j) * p + j * center_point) /
                              n3, all_coord, ctInfo, precision, shape1)
#     print(i, time()-_t)
    return all_coord


def generate_mask_sub(ctInfo, center_points, section_spl, section_u0, precision, shape1, min_sp, __i, more_sampling=False):
    _t = time()
    all_coord = []

    n = int(center_points.shape[0]/10)

    for i, ((p0, p1), (spl0, spl1), (u0, u1)) in enumerate(zip(
            pairwise(center_points),
            pairwise(section_spl),
            pairwise(section_u0))):
        n1 = np.ceil(3 * np.linalg.norm(p0 - p1) / min_sp).astype(np.int)
        n1 = max(n1, 2)
        n2 = np.ceil(
            3 * 1000 /
            min_sp * np.linalg.norm(spl0.spline(0) - spl0.spline(1 / 1000))
        ).astype(np.int)
        if more_sampling:
            n1 *= 2
            n2 = max(n2*2, 2)
            n3 = np.ceil(n2 / np.pi / 2).astype(np.int)
            n3 = max(n3, 1)

        else:
            n2 = max(n2, 2)
            n3 = np.ceil(n2 / np.pi / 2).astype(np.int)
            n3 = max(n3, 1)

        n2_linspace = np.linspace(0, 1 - 1 / n2, n2)
        next_section_points = spl1.spline(np.mod(n2_linspace+u1, 1.0))

        break
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=32 if more_sampling else 1) as executor:
        for i, ((p0, p1), (spl0, spl1), (u0, u1)) in enumerate(zip(
                pairwise(center_points),
                pairwise(section_spl),
                pairwise(section_u0))):

            this_section_points = next_section_points
            next_section_points = spl1.spline(np.mod(n2_linspace+u1, 1.0))
            for i in range(n1):
                section_points = (
                    (n1 - i) * this_section_points + i * next_section_points) / n1
                center_point = ((n1 - i) * p0 + i * p1) / n1
    #             set_mask(center_point)

                results.append(executor.submit(
                    generate_mask_sub2, center_point, section_points, n3, ctInfo, precision, shape1))

        for fs in concurrent.futures.as_completed(results):
            all_coord += fs.result()
#     print(__i, time()-_t)
    return all_coord


def generate_mask_sub2(center_point, section_points, n3, ctInfo, precision, shape1):
    all_coord = []
    generate_mask_set(center_point, all_coord, ctInfo, precision, shape1)
    for p in section_points:
        #                 set_mask(p,ret)
        for j in range(n3):
            #                     pass
            #                     set_mask(((n3 - j) * p + j * center_point) / n3)
            generate_mask_set(((n3 - j) * p + j * center_point) /
                              n3, all_coord, ctInfo, precision, shape1)

    return all_coord


# In[ ]:


def save_mask(ctVolume, result, vessel_name='', save_dir='./'):
    '''
    ctVolume: helper->CtColume
    result: output from parse_results
    '''
    save_dir = Path(save_dir)

    if not save_dir.exists() or not save_dir.is_dir():
        os.makedirs(str(save_dir.resolve()))

    if vessel_name != '':
        res_data = {}
        res_data[vessel_name] = result[vessel_name]
    else:
        res_data = result
    path1 = save_dir.joinpath(vessel_name+'_mask.npy')
    path2 = save_dir.joinpath(vessel_name+'_straighten_mask.npy')
    path3 = save_dir.joinpath(vessel_name+'_straighten.npy')
    if path1.exists() and path2.exists() and path3.exists():
        print('bypass '+vessel_name)
        return

    for vessel_name, res in res_data.items():
        if not 'inner_wall' in res:
            continue
        cor = Coronary(res, cp_offset = abs(ctVolume.spacing)/2, fix_orifice=result)
        mask = generate_mask(cor, ctVolume, precision=(4, 4, 4))
        path = save_dir.joinpath(vessel_name+'_mask.npy')
        if np.isnan(mask).any():
            print('Nan noted, {} not saved!'.format(str(path.resolve())))
        else:
            if path1.exists():
                os.remove(str(path1.resolve()))

            np.save(str(path1.resolve()), mask)

        s_mask, s_mpr = straighten_data_mask(
            ctVolume, cor, output_dim=(200, 200), precision=(4, 4), output_spacing=0.1, upsample_z=4, average_z=True)
        if np.isnan(s_mask).any():
            print('Nan noted, {} not saved!'.format(str(path2.resolve())))
        else:
            if path2.exists():
                os.remove(str(path2.resolve()))

            np.save(str(path2.resolve()), s_mask)

        if np.isnan(s_mpr).any():
            print('Nan noted, {} not saved!'.format(str(path3.resolve())))
        else:
            if path3.exists():
                os.remove(str(path3.resolve()))

            np.save(str(path3.resolve()), s_mpr)

        print('{} saved!'.format(vessel_name))


# In[ ]:


def save_straightened_mask(ctVolume, result, vessel_name='', save_dir='./', precision=(1, 1)):
    save_dir = Path(save_dir)

    if not save_dir.exists() or not save_dir.is_dir():
        os.makedirs(str(save_dir.resolve()))

    if vessel_name != '':
        s_mask, s_mpr = straighten_data_mask(
            ct, result[vessel_name], output_dim=(200, 200), precision=precision, output_spacing=0.1)
        path = save_dir.joinpath(vessel_name+'_straighten_mask.npy')
        if path.exists():
            os.remove(str(path.resolve()))
        np.save(str(path.resolve()), s_mask)

        path = save_dir.joinpath(vessel_name+'_straighten.npy')
        if path.exists():
            os.remove(str(path.resolve()))
        np.save(str(path.resolve()), s_mpr)
    else:
        for vessel_name, res in result.items():
            if not 'inner_wall' in res:
                continue
            s_mask, s_mpr = straighten_data_mask(
                ct, res, output_dim=(200, 200), precision=precision, output_spacing=0.1)
            path = save_dir.joinpath(vessel_name+'_straighten_mask.npy')
            if path.exists():
                os.remove(str(path.resolve()))
            np.save(str(path.resolve()), s_mask)

            path = save_dir.joinpath(vessel_name+'_straighten.npy')
            if path.exists():
                os.remove(str(path.resolve()))
            np.save(str(path.resolve()), s_mpr)

            print('{} saved!'.format(vessel_name))


# In[ ]:


def load_mask(ct_dir, mask_dir=''):

    if type(ct_dir) is CtVolume:
        ct = ct_dir
    else:
        ct = CtVolume()
        ct.load_image_data(ct_dir)

    if mask_dir == '':
        mask_dir = Path('.').joinpath(ct.id)
        assert mask_dir.exists()
    else:
        mask_dir = Path(mask_dir)

    mask = {}
    for npy in mask_dir.glob('*.npy'):
        vessel_name = re.sub(r'(_mask)?.npy$', '', str(npy.name), re.I)
        mask[vessel_name] = np.load(str(npy))

    return ct, mask


# In[ ]:


def save_all():
    for d in Path(r'/data/scsnake/ccta/').glob(r'S*/'):

        ct_dir = ''
        res_dir = ''

        for d1 in d.glob(r'originalDATA_name*/'):
            for d2 in d1.iterdir():
                ct_dir = str(d2)
                break
            break
        for d1 in d.glob(r'centerlineDATA*/'):
            for d2 in d1.iterdir():
                res_dir = str(d2)
                break
            break

    # %%pixie_debugger
        ct = CtVolume()
        ct.load_image_data(ct_dir + '/')

        result = parse_results(res_dir + '/')

        res = [(k if 'inner_wall' in v else '_'+k) for k, v in result.items()]
        print(ct_dir)
        print(ct.id)
        print(res)

        # In[185]:

#         save_mask(ct, result, save_dir='/data/scsnake/ccta/_'+ct.id)
    #     break


# In[ ]:


# %%pixie_debugger
def verify_npy(i):
    d = sorted(list(x for x in Path(
        r'/data/scsnake/ccta/').glob(r'S*_1.2*/') if x.is_dir()))[i]
    files = sorted(list(d.glob(r'*.npy')))
    print(np.array(list(map(str, files))))
    fig, ax = plt.subplots(len(files), 1, figsize=(8, 8*len(files)))

    if len(files) == 1:
        ax = np.array([ax])

    for f, x in zip(files, ax.ravel()):
        path = str(f)
        npy = np.load(path)
        if '_straighten_mask' in path:
            npy = npy[:, np.floor(npy.shape[1]/2).astype(np.int), :]
        elif '_straighten' in path:
            npy = npy[:, np.floor(npy.shape[1]/2).astype(np.int), :]
        elif '_mask' in path:
            npy = np.max(npy, axis=0)

        x.imshow(npy, cmap='gray')
        x.set_title(f.name)

    plt.tight_layout()
    plt.show()

# verify_npy(2)


# In[1]:


def plot3d(*args, cont=False, **kwargs):
    global fig, ax, plt
    if not cont:
        fig = plt.figure(figsize=(8,8))
        ax = Axes3D(fig)
    for ps in args:
        ax.scatter(*ps.T, **kwargs)
    set_axes_equal(ax)
    plt.show()


# In[ ]:


@njit
def minmax(array):
    # Ravel the array and return early if it's empty
    array = array.ravel()
    length = array.size
    if not length:
        return

    # We want to process two elements at once so we need
    # an even sized array, but we preprocess the first and
    # start with the second element, so we want it "odd"
    odd = length % 2
    if not odd:
        length -= 1

    # Initialize min and max with the first item
    minimum = maximum = array[0]

    i = 1
    while i < length:
        # Get the next two items and swap them if necessary
        x = array[i]
        y = array[i+1]
        if x > y:
            x, y = y, x
        # Compare the min with the smaller one and the max
        # with the bigger one
        minimum = min(x, minimum)
        maximum = max(y, maximum)
        i += 2

    # If we had an even sized array we need to compare the
    # one remaining item too.
    if not odd:
        x = array[length]
        minimum = min(x, minimum)
        maximum = max(x, maximum)

    return minimum, maximum


# In[ ]:


def fill_lumen(wall_points, vessel_p, min_sp):
    z_axis = vessel_p.vec_z
    y_axis = vessel_p.vec_y
    x_axis = vessel_p.vec_x
    
    axes = np.array([y_axis, x_axis]).T
    
    cp = vessel_p.center
    rel_coord = np.matmul(wall_points - cp, axes) / min_sp
    
    rel_coord = np.floor(rel_coord+0.5).astype(np.int)
#     cp_coord = np.floor(cp_coord+0.5).astype(np.int)
    
    min_x, max_x = minmax(rel_coord[:,1])
    min_y, max_y = minmax(rel_coord[:,0])
    
    origin = np.array([min_y, min_x])
    
    rel_coord -= origin
#     cp_coord -= origin
    
    shape = (max_y-min_y+1, max_x-min_x+1)
    
    arr = coo_matrix((np.repeat(1,rel_coord.shape[0]), rel_coord.T), shape=shape, dtype=np.int).toarray()
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    try:
        cv2.floodFill(arr, None, tuple(-origin), 1, 0, 0, cv2.FLOODFILL_FIXED_RANGE)
    except:
        pass
            
    ret_coord = (np.array(np.where(arr>0)).T + origin) * min_sp
#     
    ret = np.matmul(ret_coord, np.linalg.pinv(axes)) + cp
    
    return ret

