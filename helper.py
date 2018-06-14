# -*- coding: utf-8 -*-

import copy
import csv
import glob
import os
import random
import shutil
import sys
import time
import uuid

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
import skimage as sk
from scipy.misc import imsave
from scipy.ndimage import zoom, interpolation, filters


def circle_levelset(shape, center, sqradius, scalerow=1.0):
    """Build a binary function with a circle as the 0.5-levelset."""
    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum((grid.T) ** 2, 0))
    u = np.float_(phi > 0)
    return u


class Crop_arr:
    def __init__(self, arr, corner1=None, corner2=None, crop=False):

        self.corner1 = (0, 0, 0) if corner1 is None else corner1
        self.corner2 = tuple(np.array(arr.shape)) if corner2 is None else corner2

        if not crop:
            self.arr = arr
        else:
            self.arr = arr[self.indexFrom(corner1, corner2)]

    def indexFrom(self, corner1, corner2):
        return tuple(slice(i, j) for i, j in zip(corner1, corner2))

    def indexedArr(self, corner1=None, corner2=None):
        if corner1 is None:
            corner1 = self.corner1
        if corner2 is None:
            corner2 = self.corner2
        return self.arr[self.indexFrom(corner1, corner2)]

    def bounds(self):
        return self.corner1 + self.corner2

    def image(self):
        return ((self.arr * 101).astype(np.uint8))

    def crop(self):
        coords = np.argwhere(self.arr)

        # Bounding box of non-black pixels.
        try:
            new_corner1 = coords.min(axis=0)
            new_corner2 = coords.max(axis=0) + 1  # slices are exclusive at the top
        except:
            return

        self.arr = self.indexedArr(new_corner1, new_corner2)
        self.corner1 = tuple(np.array(self.corner1) + np.array(new_corner1))
        self.corner2 = tuple(np.array(self.corner1) + np.array(new_corner2) - np.array(new_corner1))

    def new_image(self, dim=None):
        ret = np.zeros(self.corner2 if dim is None else dim, np.uint8)
        ret[self.indexFrom(self.corner1, self.corner2)] = self.arr
        return ((ret * 101).astype(np.uint8))


class Morph3D:
    def __init__(self, img, seed=(0, 0, 0), alpha=1000, sigma=5.48, smoothing=1, threshold=0.31, balloon=5):
        import morphsnakes
        self.alpha = alpha
        self.sigma = sigma
        self.smoothing = smoothing
        self.threshold = threshold
        self.balloon = balloon
        self.img = img
        self.seed = seed
        # self.gI = morphsnakes.gborders(img, alpha=alpha, sigma=sigma)
        # self.mgac = morphsnakes.MorphGAC(self.gI, smoothing=smoothing, threshold=threshold, balloon=balloon)
        # self.last_levelset = self.mgac.levelset = circle_levelset(img.shape, seed, balloon)

        self.macwe = morphsnakes.MorphACWE(img, smoothing=1, lambda1=1, lambda2=2)
        self.last_levelset = self.macwe.levelset = circle_levelset(img.shape, seed, balloon)

        self.last_crop_levelset = Crop_arr(self.last_levelset)
        self.iter_ind = 0
        self.max_balloon = balloon * 2

    def step(self, iters=1):
        balloon = self.balloon
        img = self.img

        # self.mgac.step()
        # self.last_levelset = self.mgac.levelset

        # self.macwe.step()
        # self.last_levelset = self.macwe.levelset
        # return

        # Coordinates of non-black pixels.
        coords = np.argwhere(self.last_levelset)

        # Bounding box of non-black pixels.
        try:
            x0, y0, z0 = coords.min(axis=0)
            x1, y1, z1 = coords.max(axis=0) + 1  # slices are exclusive at the top
        except:
            return
        bounds = [x0, y0, z0, x1, y1, z1]
        crop_bounds = []
        for i, c in enumerate(bounds):
            c += balloon * (iters + 1) if i > 2 else balloon * (-1) * (iters + 1)
            if i < 3:
                c = 0 if c < 0 else c
            else:
                s = img.shape[i - 3]
                c = s if c > s else c

            crop_bounds.append(c)
        # print(self.last_crop_levelset.shape)
        (x0, y0, z0, x1, y1, z1) = crop_bounds = tuple(crop_bounds)
        # gI = morphsnakes.gborders(img[x0:x1, y0:y1, z0:z1], alpha=self.alpha, sigma=self.sigma)
        # balloon = self.balloon

        macwe = morphsnakes.MorphACWE(img[x0:x1, y0:y1, z0:z1], smoothing=1, lambda1=1, lambda2=2)
        macwe.levelset = self.last_levelset[x0:x1, y0:y1, z0:z1]
        # balloon = self.balloon * random.uniform(0.5, 1.2)
        # mgac = morphsnakes.MorphGAC(gI, smoothing=self.smoothing, threshold=self.threshold, balloon=balloon)
        # mgac.levelset = self.last_levelset[x0:x1, y0:y1, z0:z1]
        for _ in range(iters):
            macwe.step()
        self.last_levelset[x0:x1, y0:y1, z0:z1] = macwe.levelset
        self.last_crop_levelset = Crop_arr(macwe.levelset, (x0, y0, z0), (x1, y1, z1))

        self.iter_ind += iters
        # self.balloon += iters
        # if self.balloon > self.max_balloon:
        #     self.balloon = self.max_balloon

    def step_max(self, estimated_diameter=None):
        i = 0
        q = []
        q.append(self.last_crop_levelset.indexedArr().sum())
        while 1:
            i += 1
            if i > 100:
                break
            last_last_crop_levelset = copy.copy(self.last_crop_levelset)
            self.step()
            q.append(self.last_crop_levelset.indexedArr().sum())

            if estimated_diameter and i > 2 and \
                    ((q[-1] > q[-2] and (q[-1] * 8 / np.pi) ** (1 / 3) / estimated_diameter > 1.2) or \
                     (q[-1] < q[-2] and (q[-1] * 8 / np.pi) ** (1 / 3) / estimated_diameter < 0.8)):
                return self.last_crop_levelset

            if int(q[-1]) == 0:
                return last_last_crop_levelset

            if i > 2 and (abs((q[-1] - q[-2]) * 1.0 / q[-2]) < 0.02 or abs((q[-1] - q[-3]) * 1.0 / q[-3]) < 0.05):
                return self.last_crop_levelset

            # if i > 4 and q[-1] < q[-2] < q[-3] < q[-4]:
            #     return


def nodule_segmentation(CtVolume, nodule_coord):
    morph = Morph3D(CtVolume.data, nodule_coord)
    mask = None
    for _ in range(3):
        nodule = morph.step_max()
        if nodule is not None:
            mask = nodule.new_image(CtVolume.data.shape).astype(np.bool)
            break

    return mask


class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        self.slices, rows, cols = X.shape
        self.ind = self.slices // 2

        self.im = ax.imshow(self.X[self.ind, :, :], cmap='gray')
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind, :, :])
        ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


def ViewCT(arr):
    global ax, fig, tracker
    fig, ax = plt.subplots(1, 1)

    if not type(arr).__module__ == np.__name__:
        arr = arr.apply_window()

    tracker = IndexTracker(ax, arr)

    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()


class CtVolume(object):
    def __init__(self, data=None, origin=None, spacing=None, lung_mask=None):
        self.window = (-600, 1500)
        self.data = np.array(data)
        self.gray = None
        self.lung_mask = lung_mask.astype(np.bool) if lung_mask is not None else None
        self.dimension_normalized = None
        self.origin = np.array(origin)
        self.spacing = np.array(spacing)
        self.nodules = []
        self.virtual_nodules_coord = []  # pixel coordinate

    def has_lung(self):
        nonzero = np.count_nonzero(self.lung_mask)
        if nonzero == 0:
            return 0
        else:
            return 1.0 * np.count_nonzero(self.lung_mask) / self.data.size

    def load_nodule_info(self, filename):
        with open(filename, 'r') as f:
            r = csv.reader(f, delimiter=',', quotechar='"')
            for row in r:
                if row[0] == self.id:
                    nodule = {}
                    nodule['coord'] = [float(row[3]), float(row[2]), float(row[1])]
                    nodule['diameter'] = float(row[4])
                    self.nodules.append(nodule)

    def load_image_data(self, filename):
        if filename.endswith('.mhd') or filename.endswith('.mha'):
            try:
                self.data, self.origin, self.spacing = self.load_itk(filename)
            except:
                pass
        else:
            try:
                self.data, self.origin, self.spacing = self.load_dicom(filename)
            except:
                pass

    def nodule_in_VOI(self, volume):
        shape, origin, spacing = volume.data.shape, volume.origin, volume.spacing

        for nodule in self.nodules:
            is_outside = True
            for i in range(3):
                is_outside &= (origin[i] + spacing[i] * shape[i] < nodule['coord'][i] - nodule['diameter'] or
                               nodule['coord'][i] + nodule['diameter'] < origin[i])

            if not is_outside:
                return True

        return False

    def load_lung_mask(self, lung_mask_path):
        ma = self.load_itk(lung_mask_path)

        self.lung_mask = ma[0].astype(np.bool)
        # self.lung_masked = np.ma.masked_array(self.data, ~self.lung_seg)

    def masked_lung(self):
        return np.ma.masked_array(self.data, ~self.lung_mask)

    def dimension_normalize(self, volume=None):
        if volume is None:
            volume = self

        min_spacing = min(volume.spacing)
        zoom_ratio = [1 if s == min_spacing else s / min_spacing
                      for s in volume.spacing]

        volume.dimension_normalized = CtVolume(zoom(volume.data, zoom_ratio), volume.origin,
                                               (min_spacing,) * len(volume.spacing),
                                               zoom(volume.lung_mask, zoom_ratio))

    def generate_negative_volume(self, shape=(64, 64, 64), normalize=True, max_try=1000):
        # if self.lung_seg is None:
        #     self.lung_segmentation()

        # if normalize and self.dimension_normalized is None:
        #     self.dimension_normalize()
        #     volume = self.dimension_normalized
        # else:
        #     volume = self
        volume = self
        data = volume.data

        dia = [int(shape[i] / 2) for i in range(3)]
        center_pixel_coord = [random.randint(dia[i], data.shape[i] - 1 - dia[i])
                              for i in range(3)]
        padding = -1024

        tried = 0

        while 1:
            tried += 1
            if tried > max_try:
                print('Failed to generate negative volume!')
                return None

            crop = self.crop(volume, center_pixel_coord, shape, padding)

            if not self.nodule_in_VOI(crop):
                return crop
            else:
                print('crop contains nodule.')

    def negative_background_sampler(self, shape, padding=None):
        lung_HU = (-900, -500)
        tried = 0
        while 1:
            tried += 1
            if tried > 1000:
                break
            if padding is None:
                dia = [int(shape[i] / 2) for i in range(3)]
                center_pixel_coord = [random.randint(dia[i], self.data.shape[i] - 1 - dia[i])
                                      for i in range(3)]
                padding = -1024
            else:
                center_pixel_coord = [random.randint(0, self.data.shape[i] - 1)
                                      for i in range(3)]
            crop = self.crop(center_pixel_coord, shape, padding)

            # if crop.data.min()>lung_HU[1] or crop.data.max()<lung_HU[0]:
            #     continue

            estimated_lung_parenchyma_ratio = ((lung_HU[0] < crop.data) & (crop.data < lung_HU[1])).sum() * 1.0 / (
                    shape[0] * shape[1] * shape[2]
            )
            if estimated_lung_parenchyma_ratio < 0.5:
                continue
            # ViewCT(crop.data)
            if self.nodule_in_VOI(crop):
                continue
            # print(tried, center_pixel_coord)
            return crop
        return None

    def load_itk(self, filename):
        # Reads the image using SimpleITK
        itkimage = sitk.ReadImage(filename)
        self.id = os.path.splitext(os.path.basename(filename))[0]
        # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
        ct_scan = sitk.GetArrayFromImage(itkimage)
        # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
        origin = np.array(list(reversed(itkimage.GetOrigin())))

        # Read the spacing along each dimension
        spacing = np.array(itkimage.GetSpacing()).reshape((1, 3))
        direction = np.array(itkimage.GetDirection()).reshape((3, 3))
        spacing = np.dot(spacing, direction).reshape(3)[::-1]
        return ct_scan, origin, spacing

    def load_dicom(self, dirname):

        dcm_files = glob.glob(os.path.join(dirname, '[!_]*.dcm'))
        if len(dcm_files) == 0:
            dcm_files = glob.glob(os.path.join(dirname, '[!_]*'))
        assert len(dcm_files) > 0
        self.id = os.path.basename(dirname)
        f = dicom.read_file(dcm_files[0])

        s, i = f.RescaleSlope * 1.0, f.RescaleIntercept
        # p = f.pixel_array

        data = np.zeros((len(dcm_files), f.Columns, f.Rows), np.int16)
        spacing = np.array([f.SliceThickness, f.PixelSpacing[1], f.PixelSpacing[0]])
        origin = reversed(f[0x20, 0x32].value)  # image position

        spacing = np.dot(spacing[::-1].reshape((1, 3)),
                         np.append(np.array(f.ImageOrientationPatient), [0, 0, 1]).reshape((3, 3))).reshape(3)[::-1]

        for ind, path in enumerate(dcm_files):
            f = dicom.read_file(path)

            p = f.pixel_array
            data[ind] = np.array(p)

        return (data * s + i).astype(np.int16), origin, spacing

        # self.gray = self.apply_window(self.data)

    def pixel_to_absolute_coord(self, coord):
        '''
        :param coord: zero-based pixel coord x, y, z (slice number)
        :return:
        '''
        return self.origin + self.spacing * np.array(coord)

    def absolute_to_pixel_coord(self, coord):
        return np.abs(np.floor((np.array(coord) - self.origin) / self.spacing + 0.5)).astype(np.uint16)

    def apply_window(self, data=None, window=None):
        if data is None:
            data = np.copy(self.data)

        wl, ww = self.window if window is None else window

        data[data < wl - ww] = wl - ww
        data[data >= wl + ww] = wl + ww - 1
        return ((data - (wl - ww)) * 256.0 / (2 * ww)).astype(np.uint32)

    def crop(self, volume, center_pixel_coord, shape, padding):
        if type(shape) is int:
            shape = (shape,) * 3

        ret = np.full(shape, padding)
        ret_mask = np.full(shape, False)
        data_shape = volume.data.shape

        r1 = np.zeros((3, 2), np.int16)  # source range
        r2 = np.zeros((3, 2), np.int16)  # destination range
        origin = []
        for i in range(3):
            if shape[i] % 2 == 0:
                len = shape[i] / 2
                i1 = center_pixel_coord[i] - len + 1
                i2 = center_pixel_coord[i] + len
            else:
                len = (shape[i] - 1) / 2
                i1 = center_pixel_coord[i] - len
                i2 = center_pixel_coord[i] + len

            if i1 < 0:
                r1[i, 0] = 0
                r2[i, 0] = abs(i1)
            else:
                r1[i, 0] = i1
                r2[i, 0] = 0

            if i2 > data_shape[i] - 1:
                r1[i, 1] = data_shape[i]
                r2[i, 1] = shape[i] - (i2 - data_shape[i] + 1)
            else:
                r1[i, 1] = i2 + 1
                r2[i, 1] = shape[i]

            origin.append(volume.origin[i] + (r1[i, 0] - r2[i, 0]) * volume.spacing[i])

        ret[r2[0, 0]:r2[0, 1], r2[1, 0]:r2[1, 1], r2[2, 0]:r2[2, 1]] = \
            volume.data[r1[0, 0]:r1[0, 1], r1[1, 0]:r1[1, 1], r1[2, 0]:r1[2, 1]]
        if volume.lung_mask:
            ret_mask[r2[0, 0]:r2[0, 1], r2[1, 0]:r2[1, 1], r2[2, 0]:r2[2, 1]] = \
                volume.lung_mask[r1[0, 0]:r1[0, 1], r1[1, 0]:r1[1, 1], r1[2, 0]:r1[2, 1]]

            return CtVolume(ret, origin, volume.spacing, ret_mask)
        else:
            return CtVolume(ret, origin, volume.spacing)

    def save_mhd(self, data_path, mask_path=None):
        mhd = sitk.GetImageFromArray(self.data)
        mhd.SetOrigin(self.origin)
        mhd.SetSpacing(self.spacing)

        sitk.WriteImage(mhd, data_path)

        if mask_path is not None:
            mhd = sitk.GetImageFromArray(self.lung_mask.astype(np.int))
            mhd.SetOrigin(self.origin)
            mhd.SetSpacing(self.spacing)

            sitk.WriteImage(mhd, mask_path)


class Nodule(CtVolume):
    def __init__(self, image_data, image_mask=None):
        super(Nodule, self).__init__()

        if type(image_data).__module__ == np.__name__:
            self.data = image_data
        else:
            self.load_image_data(image_data)

        if type(image_mask).__module__ == np.__name__:
            self.mask = image_mask
        else:
            self.mask, origin, spacing = self.load_itk(image_mask) if image_mask else (None, None, None)
        # self.mask = zoom(self.mask, (self.data.shape[0] * 1.0 / self.mask.shape[0], 1, 1))

        if not self.data_mask_match():
            raise Exception('data/mask not match')

        if self.data.shape != self.mask.shape:
            min_spaicng = min(self.spacing)
            self.data = zoom(self.data, (self.mask.shape[0] * 1.0 / self.data.shape[0], 1, 1))
            self.spacing = (min_spaicng,) * len(self.spacing)

        self.masked_nodule_data = None
        # print(self.origin, self.spacing)
        # print(origin, spacing)

    def data_mask_match(self):
        try:
            s1 = self.data.shape
            s2 = self.mask.shape
            if s1[1] == s2[1] and s1[2] == s2[2]:
                return True
            else:
                return False
        except:
            return False

    def masked_data(self, data=None, mask=None, threshold=-3.2, fill=-1024):
        if data is None:
            data = self.data
        if mask is None:
            mask = self.mask
        # mask = np.full(self.mask.shape, True, np.bool)

        if mask.dtype == np.bool:
            m = ~mask
        else:
            m = mask < threshold
        # return self.crop_image(data, fill)

        coords = np.argwhere(~m)

        # Bounding box of non-black pixels.
        x0, y0, z0 = coords.min(axis=0)
        x1, y1, z1 = coords.max(axis=0) + 1  # slices are exclusive at the top
        try:
            ma = np.ma.masked_array(data[x0:x1, y0:y1, z0:z1], m[x0:x1, y0:y1, z0:z1])
        except:
            print(data)
            print(mask)
            sys.exit(1)
        # self.masked_nodule_data = ma
        return ma

    def rotate_random(self, nodule=None):
        if nodule is None:
            nodule = self
            d, m = nodule.data, nodule.mask
        else:
            d, m = np.copy(nodule.data), np.copy(nodule.mask)

        angle = random.uniform(0.0, 360.0)
        axes = tuple(random.sample(range(3), 2))
        d = interpolation.rotate(d, angle, axes, cval=-1024)

        padding = -4.0 if m.dtype != np.bool else False
        m = interpolation.rotate(m, angle, axes, cval=padding)

        if nodule == self:
            self.data = d
            self.mask = m
            return self
        else:
            ret = copy.deepcopy(self)
            ret.data = d
            ret.mask = m

            return ret

    def zoom_random(self, nodule=None, zoom_range=(0.3, 1.0)):
        if nodule is None:
            nodule = self
            d, m = nodule.data, nodule.mask
        else:
            d, m = np.copy(nodule.data), np.copy(nodule.mask)

        zoom_ratio = [random.uniform(*zoom_range) for _ in range(3)]
        d = interpolation.zoom(d, zoom_ratio)
        m = interpolation.zoom(m, zoom_ratio)

        if nodule == self:
            self.data = d
            self.mask = m
            return self
        else:
            ret = copy.deepcopy(self)
            ret.data = d
            ret.mask = m

            return ret

    def noise_random(self, nodule=None):
        if nodule is None:
            nodule = self
            d, m = nodule.data, nodule.mask
        else:
            d, m = np.copy(nodule.data), np.copy(nodule.mask)

        ma = self.masked_data(d, m)
        # mean = np.ma.average(ma)
        std = np.ma.std(ma)

        d += np.random.normal(0, std / 20.0, d.shape).astype(d.dtype)

        if nodule == self:
            self.data = d
            self.mask = m
            return self
        else:
            ret = copy.deepcopy(self)
            ret.data = d
            ret.mask = m

    def crop_image(self, img, tol=0):
        mask = img > tol

        # Coordinates of non-black pixels.
        coords = np.argwhere(mask)

        # Bounding box of non-black pixels.
        x0, y0, z0 = coords.min(axis=0)
        x1, y1, z1 = coords.max(axis=0) + 1  # slices are exclusive at the top

        # Get the contents of the bounding box.
        return img[x0:x1, y0:y1, z0:z1]


def nodule_blender(nodule_volume, background_volume):
    nodule_masked_array = nodule_volume.masked_data()
    nd_regions = sk.measure.regionprops(sk.measure.label(~np.ma.getmask(nodule_masked_array)))
    diameter = np.array([nd_regions[0].bbox[i + 3] - nd_regions[0].bbox[i]
                         for i in range(3)])

    centroid = np.array(nd_regions[0].centroid).astype(np.int)
    mean_diameter = np.sqrt(np.dot(diameter, diameter) / 3.0)
    half_diameter = int(mean_diameter / 2)
    if half_diameter < 1:
        half_diameter = 1

    bg_shape = background_volume.data.shape
    lung_mask = background_volume.lung_mask

    masked_lung = background_volume.masked_lung()
    lung_mean = np.ma.mean(masked_lung)
    lung_std = np.ma.std(masked_lung)

    has_lung = np.any(lung_mask)

    if has_lung:
        larger_lung_mask = sk.morphology.binary_dilation(lung_mask,
                                                         sk.morphology.ball(half_diameter))

        larger_lung_mask_coords = np.nonzero(larger_lung_mask)
        larger_lung_mask_points = len(larger_lung_mask_coords[0])

    tried = 0
    while 1:
        tried += 1

        # if tried>100:
        #     attention=1
        if tried > 1000:
            break

        bg = np.copy(background_volume.data)

        if has_lung:
            ind = random.randint(0, larger_lung_mask_points - 1)
            cr = [larger_lung_mask_coords[i][ind] for i in range(3)]
        else:
            cr = [random.randint(0 - half_diameter, bg_shape[i] + half_diameter - 1)
                  for i in range(3)]
        new_nd_mask = np.full(bg_shape, False)

        overlap_point_count = 0
        all_point_count = 0
        for point in nd_regions[0].coords:
            coord = [cr[i] + point[i] for i in range(3)]
            if not (0 <= coord[0] < bg_shape[0] and 0 < coord[1] < bg_shape[1] and 0 <= coord[2] < bg_shape[2]):
                continue
            nd_HU = nodule_masked_array[tuple(point)]
            bg_HU = bg[tuple(coord)]

            if bg_HU > lung_mean and (bg_HU > nd_HU or abs(nd_HU - bg_HU) < lung_std):
                overlap_point_count += 1
            all_point_count += 1
            bg[tuple(coord)] = max(nd_HU, bg_HU)
            new_nd_mask[tuple(coord)] = True

        new_nd_centroid = [cr[i] + centroid[i] for i in range(3)]

        if all_point_count == 0:
            continue

        overlap_ratio = overlap_point_count * 1.0 / all_point_count

        visible_point_count = all_point_count - overlap_point_count
        shown_ratio = visible_point_count * 1.0 / nd_regions[0].area

        if visible_point_count > 100 or shown_ratio > 0.5:
            # if overlap_ratio < 0.7 and shown_ratio > 0.3:
            # new_nd_out_margin=filters.convolve(new_nd_mask.astype(np.int), np.array([[[0,0,0], [0,1,0], [0,0,0]],
            #                                         [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            #                                         [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]))
            # new_nd_out_margin = (new_nd_out_margin > 0) & (~new_nd_mask)

            new_nd_out_margin = sk.morphology.binary_dilation(new_nd_mask,
                                                              sk.morphology.ball(random.randint(1, 2)))
            new_nd_out_margin ^= new_nd_mask

            gau_blur = filters.gaussian_filter(bg, random.uniform(0.3, 1.5))

            new_nd_out_margin_coords = np.nonzero(new_nd_out_margin)

            portion = random.uniform(0.1, 0.4)
            bg[new_nd_out_margin_coords] = bg[new_nd_out_margin_coords] * portion + \
                                           gau_blur[new_nd_out_margin_coords] * (1.0 - portion)

            if new_nd_centroid[0] >= bg_shape[0]:
                # thumbnail = np.reshape(bg[-1, :, :], (1, bg_shape[1], bg_shape[2]))
                thumbnail = bg[-1, :, :]
            elif new_nd_centroid[0] <= 0:
                # thumbnail = np.reshape(bg[0, :, :], (1, bg_shape[1], bg_shape[2]))
                thumbnail = bg[0, :, :]
            else:
                # thumbnail = np.reshape(bg[new_nd_centroid[0], :, :], (1, bg_shape[1], bg_shape[2]))
                thumbnail = bg[new_nd_centroid[0], :, :]

            # return thumbnail
            output = copy.deepcopy(background_volume)
            output.data = bg
            # output.lung_mask &= ~new_nd_mask
            output.nodule_mask = new_nd_mask

            bbox_coord1 = [cr[i] + nd_regions[0].bbox[i] for i in range(3)]
            bbox_coord2 = [cr[i] + nd_regions[0].bbox[i + 3] for i in range(3)]
            output.virtual_nodules_coord.append(tuple(bbox_coord1 + bbox_coord2))

            return output, thumbnail

    return None, None


def data_prepare(dataset_dir, dataset_nodule_csv, nodule_dir,
                 output_dir_nolung, output_dir_lung_no_nodule, output_dir_lung_nodule,
                 output_nolung_total, output_lung_no_nodule_total, output_lung_nodule_total,
                 nodule_rotate=True, nodule_zoom=True):
    for d in [output_dir_nolung, output_dir_lung_no_nodule, output_dir_lung_nodule]:
        if not os.path.exists(d):
            os.mkdir(d)
        if not os.path.exists(os.path.join(d, 'thumbnail')):
            os.mkdir(os.path.join(d, 'thumbnail'))

    dataset_files = glob.glob(os.path.join(dataset_dir, '*.mhd'))
    nodule_files = glob.glob(os.path.join(nodule_dir, '*_outputROI.mhd'))

    nolung_each = int(output_nolung_total / len(dataset_files))
    lung_no_nodule_each = int(output_lung_no_nodule_total / len(dataset_files))
    lung_nodule_each = int(output_lung_nodule_total / len(dataset_files))

    nolung_each = lung_no_nodule_each = lung_nodule_each = 100

    for lung_file in dataset_files:
        # if not '00745' in lung_file:
        #     continue

        img_name = os.path.basename(lung_file)
        file_id, _ = os.path.splitext(img_name)

        lung = CtVolume()
        lung.load_image_data(lung_file)
        # lung.load_lung_mask(os.path.join('.', 'result', file_id + '_ma.mhd'))
        lung.load_nodule_info(dataset_nodule_csv)

        print(lung.id + ': ' + str(len(lung.nodules)))
        print(lung.nodules)

        for nd_ind, nd in enumerate(lung.nodules):
            # if nd_ind==0:
            #     continue
            coord = nd['coord']

            dia = int(np.rint(nd['diameter']))

            # if dia > 20:
            #     continue
            coord2 = lung.absolute_to_pixel_coord(coord)
            crop = lung.crop(lung, coord2, dia + 20, -1024)
            # print(coord2)
            # ViewCT(crop)
            ballon_size = 5 if dia > 8 else int(dia / 1.5)
            rg_success = False
            for _ in range(10):
                random_int = random.randint(-2, 2) if dia > 8 else random.randint(-1, 1)
                rg = Morph3D(crop.data, crop.absolute_to_pixel_coord(coord), balloon=ballon_size + random_int).step_max(
                    dia)

                if rg and 0.4 < (rg.indexedArr().sum() * 8 / np.pi) ** (1 / 3) / dia < 1.5:
                    print('saving nodule #' + str(nd_ind))
                    crop.lung_mask = rg.new_image(crop.data.shape).astype(np.bool)

                    crop.save_mhd('./lung_nodule_segmentation/' + str(lung.id) + str(nd_ind).zfill(3) + '_' + str(
                        int(dia)) + '.mhd',
                                  './lung_nodule_segmentation/' + str(lung.id) + str(nd_ind).zfill(3) + '_ma.mhd')
                    crop.data = crop.masked_lung().filled(-1024)
                    crop.save_mhd('./lung_nodule_segmentation/' + str(lung.id) + str(nd_ind).zfill(3) + '_se.mhd')
                    crop.data = crop.apply_window()
                    crop.save_mhd('./lung_nodule_segmentation/' + str(lung.id) + str(nd_ind).zfill(3) + '_lu.mhd')
                    rg_success = True
                    break

            if not rg_success:
                print(
                    'Region grow error for lung id %s, #%s nodule diameter %s' % (str(lung.id), str(nd_ind), str(dia)))

        continue
        nolung_count = lung_no_nodule_count = lung_nodule_count = 0

        while 1:
            crop = lung.generate_negative_volume()
            # print('crop')
            if crop is None:
                continue

            has_lung = crop.has_lung()

            id = str(uuid.uuid4())

            if has_lung == 0 and nolung_count < nolung_each and False:
                while 1:
                    output_path = os.path.join(output_dir_nolung, file_id + '-' + id + '.mhd')
                    if not os.path.exists(output_path):
                        break
                    else:
                        id = str(uuid.uuid4())
                        print('Create new uid: ' + id)
                crop.save_mhd(output_path)
                imsave(os.path.join(output_dir_nolung, 'thumbnail', file_id + '-' + id + '_tn.png'),
                       crop.apply_window(crop.data[int(crop.data.shape[0] / 2), :, :]))
                nolung_count += 1
                print('Create no_lung #%d from lung %s' % (nolung_count, file_id))
            elif has_lung > 0.1:

                if lung_nodule_count < lung_nodule_each and lung_no_nodule_count < lung_no_nodule_each:
                    token1, token2 = tuple(random.sample(range(2), 2))
                else:
                    token1 = token2 = True

                # token1 , token2 = False, True
                token1, token2 = True, False

                if token2 and lung_no_nodule_count < lung_no_nodule_each:

                    while 1:

                        output_path = os.path.join(output_dir_lung_no_nodule, file_id + '-' + id + '.mhd')

                        if not os.path.exists(output_path):

                            break

                        else:

                            id = str(uuid.uuid4())

                    crop.save_mhd(output_path,

                                  os.path.join(output_dir_lung_no_nodule, file_id + '-' + id + '_mask.mhd'))

                    try:

                        regions = sk.measure.regionprops(sk.measure.label(crop.lung_mask))

                        centroid_z = int(regions[0].centroid[0])

                    except:

                        centroid_z = int(crop.data.shape[0] / 2)

                    imsave(os.path.join(output_dir_lung_no_nodule, 'thumbnail', file_id + '-' + id + '_tn.png'),

                           crop.apply_window(crop.data[centroid_z, :, :]))

                    lung_no_nodule_count += 1
                    print('Create lung_no_nodule #%d from lung %s' % (lung_no_nodule_count, file_id))
                elif token1 and lung_nodule_count < lung_nodule_each:

                    # if ~has_lung:
                    #     continue
                    # if True:
                    nd_ind = random.randint(0, len(nodule_files) - 1)
                    nd_file = nodule_files[nd_ind]

                    nd_name = os.path.basename(nd_file)
                    nd_id, _ = os.path.splitext(nd_name)
                    nd_id = nd_id[0:10]
                    try:
                        nd = Nodule(nd_file,
                                    os.path.join(nodule_dir, nd_id + '_outputTumorImage.mha'))
                    except:
                        print('Nodule load error!')
                        continue

                    if nodule_rotate:
                        nd.rotate_random()
                    if nodule_zoom:
                        nd.zoom_random()

                    # nd.noise_random()

                    print('start blending...')

                    for try_blending in range(3):
                        output, thumbnail = nodule_blender(nd, crop)

                        if output is None:
                            print('retry blending with smaller nodule ...')
                            output, thumbnail = nodule_blender(nd.zoom_random(zoom_range=(0.5, 0.7)), crop)
                            continue
                        else:
                            break

                    if output is None:
                        print('failed blending...')
                        continue

                    while 1:
                        output_path = os.path.join(output_dir_lung_nodule, file_id + '-' + id + '.mhd')
                        if not os.path.exists(output_path):
                            break
                        else:
                            id = str(uuid.uuid4())
                            print('Create new uid: ' + id)

                    output.save_mhd(output_path,
                                    os.path.join(output_dir_lung_nodule, file_id + '-' + id + '_mask.mhd'))
                    imsave(os.path.join(output_dir_lung_nodule, 'thumbnail', file_id + '-' + id + '_tn.png'),
                           thumbnail)
                    lung_nodule_count += 1
                    print('Create lung_nodule #%d from lung %s and nodule %s' % (lung_nodule_count, file_id, nd_id))



            elif nolung_count >= nolung_each and lung_no_nodule_count >= lung_no_nodule_each and lung_nodule_count >= lung_nodule_each:
                break
        break


def reset():
    try:
        shutil.rmtree(r'./lung_nodule/')
    except:
        pass
    try:
        shutil.rmtree(r'./no_lung/')
    except:
        pass
    try:
        shutil.rmtree(r'./lung_no_nodule/')
    except:
        pass


dataset_folder = ''  # dataset_folder/id/*.dcm
dataset_csv = ''  # id,x,y,z in each line of csv
dataset_coord_type = 'absolute'  # absolute or pixel
output_dim = (64, 64, 64)

if __name__ == '__main__':
    # nodule_files = glob.glob(os.path.join(r'/home/scsnake/Downloads/LSTK/', '*_outputROI.mhd'))
    # for nd in nodule_files:
    #     nd_name = os.path.basename(nd)
    #     nd_id, _ = os.path.splitext(nd_name)
    #     nd_id = nd_id[0:10]
    #
    #     n=sitk.GetArrayFromImage(sitk.ReadImage(nd))
    #     m=sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(r'/home/scsnake/Downloads/LSTK/', nd_id + '_outputTumorImage.mha')))
    #
    #     s1=n.shape
    #     s2=m.shape
    #
    #     if s1[1]==s2[1] and s1[2]==s2[2]:
    #         continue
    #     else:
    #         print(s1)
    #         print(s2)

    # reset()
    start_time = time.time()
    data_prepare(r'/data/LKDS/allset/',
                 r'/data/LKDS/csv/annotations_all.csv',
                 r'/home/scsnake/Downloads/LSTK/',
                 r'./no_lung/',
                 r'./lung_no_nodule/',
                 r'./lung_nodule/',
                 100, 100, 100)
    print(time.time() - start_time)
    # d = CtVolume()

    # d.load_image_data(r'/data/LKDS/allset/LKDS-00024.mhd')
    # d.load_lung_mask(r'./result/LKDS-00024_ma.mhd')
    # ViewCT(d.masked_lung())
    # d.load_nodule_info(r'/data/LKDS/csv/annotations_reviewed_sorted.csv')
    # while 1:
    #     crop = d.generate_negative_volume(output_dim)
    #     if np.any(crop.lung_mask):
    #         ViewCT(crop.apply_window(crop.masked_lung()))
    # nd = Nodule(r'./LSTK/LKDS-00024_outputROI_ps.mhd',
    #             r'./LSTK/LKDS-00024_outputTumorImage_ps.mha')
    #
    # pickle.dump(d, open('LKDS-00024-lung','w'), pickle.HIGHEST_PROTOCOL)
    # pickle.dump(nd, open('LKDS-00024-nodule', 'w'), pickle.HIGHEST_PROTOCOL)
    # d = pickle.load(open('LKDS-00024-lung'))
    # nd = pickle.load(open('LKDS-00024-nodule'))

    # ViewCT(nd.apply_window())
    # ViewCT(nd.apply_window(nd.rotate_random()))
    # d.load_itk(r'C:\LKDS\LKDS-00001.mhd')
    # t = d.crop((200,200,200), (128, 128,128), 127)
    # t = d.nodule_blender(nd)
    # while 1:
    #     crop = d.generate_negative_volume(output_dim)
    #     t = nodule_blender(nd.noise_random(), crop)
    #
    #     # np.save('new.npy', t )
    #     if t is not None:
    #         ViewCT(t)

    # plt.axis('off')
    # for i in range(16):
    #     plt.subplot(4, 4, i + 1)
    #     plt.imshow(t[int(t.shape[0] *1.0/ 16 * i)], cmap='gray')
    #
    # plt.show()
