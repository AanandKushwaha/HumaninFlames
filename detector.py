import numpy as np 
from skimage import transform
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
import imutils
import dill as pickle
from skimage.feature import hog
from sklearn.externals import joblib
import cv2
#from config import *
from skimage import color
import matplotlib.pyplot as plt 
import os 
import glob

pos_im_path = 'persons\pos'
neg_im_path = 'persons\neg'
min_wdw_sz = [64, 128]
step_size = [10,10]
orientations = 9
pixels_per_cell = [6,6]
cells_per_block = [2, 2]
visualize = False
normalize = True
pos_feat_ph = 'features\pos_mod'
neg_feat_ph = 'features\neg_mod'
model_path = 'models'
threshold = .3



def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input 'image' of size 
    equal to 'window_size'. The first image returned top-left 
    co-ordinate (0, 0) and are increment in both x and y directions
    by the 'step_size' supplied.

    So, the input parameters are-
    image - Input image
    window_size - Size of Sliding Window 
    step_size - incremented Size of Window

    The function returns a tuple -
    (x, y, im_window)
    '''
    for y in xrange(0, image.shape[0], step_size[1]):
        for x in xrange(0, image.shape[1], step_size[0]):
            yield (x, y, image[y: y + window_size[1], x: x + window_size[0]])

def detector(filename):
    im = cv2.imread(filename)
    im = imutils.resize(im, width = min(400, im.shape[1]))
    min_wdw_sz = (64, 128)
    step_size = (10, 10)
    downscale = 1.6
    #with open('svm.model', 'rb') as f:
     #    loaded = pickle.load(f)
         
    
    clf = joblib.load(os.path.join(model_path,'svm.model'))

    #List to store the detections
    detections = []
    #The current scale of the image 
    scale = 0

    for im_scaled in pyramid_gaussian(im, downscale = downscale):
        #The list contains detections at the current scale
        if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
            break
        for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                continue
            im_window = color.rgb2gray(im_window)
            fd =  hog(im_window, orientations=9, pixels_per_cell=(6, 6),cells_per_block=(2, 2),block_norm='L1', visualise=False,transform_sqrt=False,feature_vector=True,normalise=None)
            fd = fd.reshape(1, -1)
            pred = clf.predict(fd)

            if pred == 1:
                
                if clf.decision_function(fd) > 0.5:
                    detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), clf.decision_function(fd), int(min_wdw_sz[0] * (downscale**scale)),int(min_wdw_sz[1] * (downscale**scale))))
                 

            
        scale += 1

    clone = im.copy()



    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
    sc = [score[0] for (x, y, score, w, h) in detections]
    print ("sc: ", sc)
    sc = np.array(sc)
    pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.3)
    #print ("shape, ", pick.shape)
    if sc.any() < 0.5:
       clone=cv2.putText(clone,"Human not Detected", (100,180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0,255), lineType=cv2.LINE_AA)
       im=cv2.putText(im,"Human not Detected", (100,180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), lineType=cv2.LINE_AA)
    else:
        clone=cv2.putText(clone,"Human Detected", (100,180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), lineType=cv2.LINE_AA)
        im=cv2.putText(im,"Human Detected", (100,180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), lineType=cv2.LINE_AA)
   
        for (x_tl, y_tl, _, w, h) in detections:
            cv2.rectangle(im, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 255, 0), thickness = 2)
        for(xA, yA, xB, yB) in pick:
           cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 255, 0), 2)
    
    plt.axis("off")
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.title("before NMS")
    plt.show()

    plt.axis("off")
    plt.imshow(cv2.cvtColor(clone, cv2.COLOR_BGR2RGB))
    plt.title("after NMS")
    plt.show()

def test_folder(foldername):

    filenames = glob.glob(os.path.join(foldername, '*'))
    for filename in filenames:
        detector(filename)

if __name__ == '__main__':
    foldername = 'test_image'
    test_folder(foldername)
