import OpenEXR
import array
import Imath
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import shutil
import IO as io

def exr2flow(exr, shape=(600, 800), h=600, w=800):
    """
    Returns a 2 channel image where the first channel is the disparity in X-direction and sencond channel is the
    disparity in the Y-direction.
    
    Convention:
    pixel moves right +ve flow in X-direction
    pixel moves down +ve flow in Y-direction

    :param exr: path to exr file
    :param shape of the exr files
    :return: flow
    """
    file = OpenEXR.InputFile(exr)

    # Compute the size
    dw = file.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    (R, G, B) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B")]


    img = np.zeros((h, w, 2), np.float64)
    if np.max(-np.array(R).reshape(img.shape[0], -1)) < 0.001 and np.min(-np.array(R).reshape(img.shape[0], -1)) > -0.001:
        img[:, :, 0] = np.round(-np.array(R).reshape(img.shape[0], -1))
    else:
        img[:, :, 0] = -np.array(R).reshape(img.shape[0], -1)
    if np.max(-np.array(G).reshape(img.shape[0], -1)) < 0.001 and np.min(-np.array(G).reshape(img.shape[0], -1)) > -0.001:
        img[:, :, 1] = np.round(np.array(G).reshape(img.shape[0], -1))
    else:
        img[:, : , 1] = np.array(G).reshape(img.shape[0], -1)
      
    # plt.imshow(img[:,:,1],cmap="gray")
    # plt.title("sadfasfd")
    # plt.show()
    io.write(exr[:-4]+".flo",img)
    return img
