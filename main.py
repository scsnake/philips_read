import pydicom
from pathlib import Path
from glob import glob
from collections import OrderedDict
import numpy as np
import bpy


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

        ret[vessel_name] = np.array(center_infos).reshape((9, -1))

    return ret


if __name__ == '__main__':
    centerlines = get_centerlines(r'D:\Users\ntuhuser\Downloads\Cad\STATE0 - 805')
    a = centerlines['LAD']
    l = []
    for i in range(a.shape[1]):
        l += list(a[3:, i])

    print(l)
