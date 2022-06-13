# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 16:21:24 2022

@author: jymcl
"""

import numpy as np
from PIL import Image
img = Image.open(r"D:\Documents\Data\Martelloscope\test\healthy\DJI_0728.JPG")
imgArray = np.asarray(img)
imgArray2 = np.asarray(img).T
print(imgArray.shape)