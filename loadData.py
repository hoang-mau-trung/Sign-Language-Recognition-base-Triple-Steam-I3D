import cv2 
import numpy as np 
import os 
from videoto3D import Videoto3D
'''
part : full   load all data train and test
part : train   just load data train
part : test  just load data tes

mode : RGB input
mode : OPT inpt
'''
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

CLASSES = {'Basketball' : 1, 'BasketballDunk' : 2,'Biking' : 3,'CliffDiving': 4,'CricketBowling' : 5, 
'Diving':6 ,'Fencing' :7 ,'FloorGymnastics':8 ,'GolfSwing':9, 'HorseRiding':10,  'IceDancing':11,       
'LongJump':12, 'PoleVault':13, 'RopeClimbing':14, 'SalsaSpin' :15, 'SkateBoarding' :16,  'Skiing':17,      
'Skijet' :18  ,'SoccerJuggling':19,'Surfing':20,  'TennisSwing':21, 'TrampolineJumping':22,'VolleyballSpiking':23,
'WalkingWithDog':24 }


def load(part = "full", mode = 'rgb', number_frame =  10 ,width = 128, height = 128 ):


    folder_text  = '/content/drive/My Drive/ActionRecognition/UCF101_Action_detection_splits'
   
    folder_video = '/content/drive/My Drive/ActionRecognition/UCF101/UCF-101'

    
    ''' 3 file train and 3 file test
        train have form : trainlist0x.txt  [x for x in range(1, 3) ]
        test have form  : testlist0x.text    
    '''
    X_train = []  # data
    y_train = []  # label
    X_test  = []
    y_test  = []
    if mode == 'rgb':
        videoto3D = Videoto3D(width, height, number_frame)
    else :
        videoto3D = Videoto3D(width, height, number_frame+1)

    print('[INFOR] : Starting loading...... (^-^)')
    for i in range(3,4):
        if part == "full":
            traintxt = 'trainlist0'+ str(i) +'.txt'
            testtxt  = 'testlist0'+ str(i) +'.txt'
            path_traintxt = os.path.join(folder_text, traintxt)
            path_testtxt  = os.path.join(folder_text,testtxt)
            traintxt = load_doc(path_traintxt)
            testtxt  = load_doc(path_testtxt)
            for line in traintxt.split('\n'):
                list_split = line.split(' ')
                path_file = list_split[0]
                print("[INFOR]: Loading....", path_file.split('/'))
                label =  CLASSES[path_file.split('/')[0]]
                path_file =  os.path.join(folder_video,path_file)
                if mode == 'rgb':
                    tmp = videoto3D.video3D(path_file, True)
                    
                else:
                    tmp = videoto3D.video3D(path_file, False)
                if tmp is not None:
                    
                    X_train.append(tmp)
                    y_train.append(label)

            for line in testtxt.split('\n'):
                label = CLASSES[line.split('/')[0]]
                print("[INFOR]: Loading.....", line)
                path_file =  os.path.join(folder_video,line.split(" ")[0])
                if mode == 'rgb':

                    tmp = videoto3D.video3D(path_file, True)
                    
                else:
                    tmp = videoto3D.video3D(path_file, False)

                if tmp is not None:
                      
                    X_test.append(tmp)
                    y_test.append(label)

        elif part == 'train':
            traintxt = 'trainlist0'+ str(i) +'.txt'
            path_traintxt = os.path.join(folder_text, traintxt)
            traintxt = load_doc(path_traintxt)
            for line in traintxt.split('\n'):
                list_split = line.split(' ')
                path_file = list_split[0]
                print("[INFOR]: Loading....", path_file.split('/'))
                label =  CLASSES[path_file.split('/')[0]]
                path_file =  os.path.join(folder_video,path_file)
                if mode == 'rgb':
                    tmp = videoto3D.video3D(path_file, True)
                    
                else:
                    tmp = videoto3D.video3D(path_file, False)
                if tmp is not None:
                    
                    X_train.append(tmp)
                    y_train.append(label)
        else:
           
            testtxt  = 'testlist0'+ str(i) +'.txt'
            path_testtxt  = os.path.join(folder_text,testtxt)
            testtxt  = load_doc(path_testtxt)
            for line in testtxt.split('\n'):
                label = CLASSES[line.split('/')[0]]
                print("[INFOR]: Loading.....", line)
                path_file =  os.path.join(folder_video,line.split(" ")[0])
                if mode == 'rgb':
                    tmp = videoto3D.video3D(path_file, True)
                    
                else:
                    tmp = videoto3D.video3D(path_file, False)

                if tmp is not None:
                    
                    X_test.append(tmp)
                    y_test.append(label)
                    
    X_train = np.asanyarray(X_train)
    X_test  = np.asanyarray(X_test)
    print('[INFOR]: loaded data !')

    X_train[X_train > 20] = 20
    X_train[X_train < -20] = -20
    X_train = X_train/20
    
    X_test[X_test > 20] = 20
    X_test[X_test < -20] = -20
    X_test = X_test/20
    print('[INFOR] : Shape data for train: ', X_train.shape)
    print('[INFOR] : Shape data for test: ', X_test.shape)

    print('[INFOR] : (-_-) GOOD LUCKY FOR YOU HAVE HIGH ACCURARY  (-_-)')
    return (X_train, y_train, X_test, y_test)




