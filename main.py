import pydicom
from pathlib import Path
from glob import glob
from collections import OrderedDict
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
    np.set_printoptions(formatter={'float_kind':  lambda x: "%.2f" % x})

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
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=np.random.rand(3,))
    # ax.scatter(data2[:, 0], data2[:, 1], data2[:, 2], color='red')
    for angle in range(0, 360, 10):
        ax.view_init(30, angle)
        # plt.draw()
        plt.savefig('coronary%d.png' % (angle, ))
        # plt.pause(.001)