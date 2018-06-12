import pydicom
from pathlib import Path
from glob import glob
from collections import OrderedDict
import numpy as np

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
    centerlines = get_centerlines(r'D:\Users\ntuhuser\Downloads\Cad\STATE0 - 805')
    a = centerlines['LAD']
    l = []

    for i in range(a.shape[1]):
        l+= list(a[i, 0:3])

    print(l)
