import cv2
import numpy as np 

# from tfoptflow.extract_optFlow import Img2Flow
# from extract_optFlow import Img2Flow

from opticalflow import OpticalFlow  # Open CV >= 4
from opticalflowCv3 import OpticalFlow # OpenCV >= 3 and <= 4
class Videoto3D:

    def __init__(self,  width , height , depth = 10):

        self.width = width
        self.height = height
        self.depth = depth
        self.flower = OpticalFlow()
    
    def video3D(self, filename, color=False, skip=True):
        # color True : RGB
        # color false: hardwired kenerl 
        #skip: True division give 10 frame
        
        cap = cv2.VideoCapture(filename)

        nframe = cap.get(cv2.CAP_PROP_FRAME_COUNT) # give n frame

        if (color == True):
            frames = [x * nframe / self.depth for x in range(self.depth)]
            # print(len(frames))
        else:
            frames = [x * nframe / self.depth for x in range(self.depth)]

        framearray = []
        
        for i in range(self.depth):
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
            ret, prvs = cap.read()
            
            if prvs is None:
                break
            prvs = cv2.resize(prvs, (self.height, self.width))
            
            if color:
                framearray.append(prvs)
                
            else:
                if i == self.depth - 1:
                    break
                
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i + 1])
                    ret, next_frame = cap.read()
                    next_frame = cv2.resize(next_frame, (self.height, self.width))
                    
                    image1 = prvs
                    image2 = next_frame
                   
                    flow = self.flower.predict(image1, image2)
                    # flow = np.asanyarray(flow)[0]
                    framearray.append(flow)
                    
        cap.release()
        framearray = np.asanyarray(framearray)
        return framearray
