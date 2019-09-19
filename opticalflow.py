import os
import glob
import sys
import time
import random
import warnings
import numpy as np
import pandas as pd
import cv2

class OpticalFlow:

    def __init__(self, sAlgorithm:str = "tvl1-fast", bThirdChannel:bool = False, fBound:float = 20.):
        self.bThirdChannel = bThirdChannel
        self.fBound = fBound
        

        if sAlgorithm == "tvl1-fast":
            self.oTVL1 = cv2.optflow.DualTVL1OpticalFlow_create(
                scaleStep = 0.5, warps = 3, epsilon = 0.02)
                # Mo 25.6.2018: (theta = 0.1, nscales = 1, scaleStep = 0.3, warps = 4, epsilon = 0.02)
                # Very Fast (theta = 0.1, nscales = 1, scaleStep = 0.5, warps = 1, epsilon = 0.1)
            sAlgorithm = "tvl1"

        elif sAlgorithm == "tvl1-warps1":
            self.oTVL1 = cv2.optflow.DualTVL1OpticalFlow_create(warps = 1)
            sAlgorithm = "tvl1"

        elif sAlgorithm == "tvl1-quality":
            self.oTVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
                # Default: (tau=0.25, lambda=0.15, theta=0.3, nscales=5, warps=5, epsilon=0.01, 
                #innnerIterations=30, outerIterations=10, scaleStep=0.8, gamma=0.0, 
                #medianFiltering=5, useInitialFlow=False)
            sAlgorithm = "tvl1"

        elif sAlgorithm == "farnback":
            pass

        else: raise ValueError("Unknown optical flow type")
        
        self.sAlgorithm = sAlgorithm

        return

    def predict(self, img1, img2):
        
        h, w, _ = img1.shape
        ImgPrev = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        ImgCurrent =  cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        if self.sAlgorithm == "tvl1":
            arFlow = self.oTVL1.calc(ImgPrev, ImgCurrent, None)
        elif self.sAlgorithm == "farnback":
            arFlow = cv2.calcOpticalFlowFarneback(ImgPrev, ImgCurrent, flow=None, 
                pyr_scale=0.5, levels=1, winsize=15, iterations=2, poly_n=5, poly_sigma=1.1, flags=0)
        else: raise ValueError("Unknown optical flow type")

        # only 2 dims
        arFlow = arFlow[:, :, 0:2]

        # truncate to +/-15.0, then rescale to [-1.0, 1.0]
        # arFlow[arFlow > self.fBound] = self.fBound 
        # arFlow[arFlow < -self.fBound] = -self.fBound
        # arFlow = arFlow / self.fBound
        arZeros = np.zeros((h, w, 1), dtype = np.float32)
        if self.bThirdChannel:
            # add third empty channel
            arFlow = np.concatenate((arFlow, arZeros), axis=2) 
        return arFlow
        
# from videoto3Dv1 import Videoto3D
# videoto3D = Videoto3D(224, 224, 64)
# tmp = videoto3D.video3D("test.avi", True)
# flower = OpticalFlow() 
# flow = flower.predict(tmp[0], tmp[1])
# print(flow.shape)
# print(flow)
# print(np.max(flow))
# print(np.min(flow))


