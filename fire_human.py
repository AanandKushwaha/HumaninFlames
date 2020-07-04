# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 23:00:16 2019

@author: A7001
"""

import numpy as np 
#import joblib
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
import imutils
from skimage.feature import hog
from sklearn.externals import joblib
import cv2

from skimage import color

import os 


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
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y: y + window_size[1], x: x + window_size[0]])


            
model_path = 'models'            
clf = joblib.load(os.path.join(model_path, 'svm.model'))
cap=cv2.VideoCapture('perf.webm')
while True:
    ret,frame=cap.read()    
    im = imutils.resize(frame, width = min(400, frame.shape[1]))
    min_wdw_sz = (64, 128)
    step_size = (10, 10)
    downscale = 1.25
   
    img1=frame
    rchannel = img1[:, :, 2]
    gchannel = img1[:, :, 1]
    bchannel = img1[:, :, 0]
    
   
    Ydash = 16+(0.2567890625  * rchannel)+ (0.50412890625 * gchannel) +( 0.09790625 * bchannel)
    Cb= 128+(-0.14822265625 * rchannel)-(0.2909921875 * gchannel) + (0.43921484375* bchannel)
    Cr = 128+(0.43921484375  * rchannel)- (0.3677890625 * gchannel) -( 0.07142578125 * bchannel)
    
    Ymean=np.mean(np.mean(Ydash))
    Cbmean=np.mean(np.mean(Cb))
    Crmean=np.mean(np.mean(Cr))
    Crstd=np.std(Cr)
    
    #rule one starts here
    R1r, R1c= np.where(Ydash > Cb)
    ruleIpixel=len(R1r)
    Ir1= np.zeros(np.shape(img1), dtype='uint8')
    for i in range(ruleIpixel):
        Ir1[R1r[i],R1c[i],2] =rchannel[R1r[i],R1c[i]]
        Ir1[R1r[i],R1c[i],1] =gchannel[R1r[i],R1c[i]]
        Ir1[R1r[i],R1c[i],0] =bchannel[R1r[i],R1c[i]]
        i=i+1
    
    #rule 2 starts here
    R2r, R2c= (np.where((Ydash>Ymean) & (Cr> Crmean)))
    ruleIIpixel=len(R2r)
    Ir2=  np.zeros(np.shape(img1), dtype='uint8')
    for i2 in range(ruleIIpixel):
        Ir2[R2r[i2],R2c[i2],2] =rchannel[R2r[i2],R2c[i2]]
        Ir2[R2r[i2],R2c[i2],1] =gchannel[R2r[i2],R2c[i2]]
        Ir2[R2r[i2],R2c[i2],0] =bchannel[R2r[i2],R2c[i2]]
        i2=i2+1
        
    #rule one and two
    R12r, R12c= np.where((Ydash>Cb) & (Ydash>Ymean) & (Cr > Crmean))
    ruleI_IIpixel=len(R12r)
    Ir12= np.zeros(np.shape(img1), dtype='uint8')
    for i3 in range(ruleI_IIpixel):
        Ir12[R12r[i3],R12c[i3],2] =rchannel[R12r[i3],R12c[i3]]
        Ir12[R12r[i3],R12c[i3],1] =gchannel[R12r[i3],R12c[i3]]
        Ir12[R12r[i3],R12c[i3],0] =bchannel[R12r[i3],R12c[i3]]
        i3=i3+1
    
    #rule three
    R3r, R3c= np.where((Ydash>Cr) & (Cb>Cr) )
    ruleIIIpixel=len(R3r)
    Ir3= np.zeros(np.shape(img1), dtype='uint8')
    for i4 in range(ruleIIIpixel):
        Ir3[R3r[i4],R3c[i4],2] =rchannel[R3r[i4],R3c[i4]]
        Ir3[R3r[i4],R3c[i4],1] =gchannel[R3r[i4],R3c[i4]]
        Ir3[R3r[i4],R3c[i4],0] =bchannel[R3r[i4],R3c[i4]]
        i4=i4+1
    
    #rule four
    R4r, R4c= np.where((Cr<(7.4*Crstd)))
    ruleIVpixel=len(R4r)
    Ir4=  np.zeros(np.shape(img1), dtype='uint8')
    for i5 in range(ruleIVpixel):
        Ir4[R4r[i5],R4c[i5],2] =rchannel[R4r[i5],R4c[i5]]
        Ir4[R4r[i5],R4c[i5],1] =gchannel[R4r[i5],R4c[i5]]
        Ir4[R4r[i5],R4c[i5],0] =bchannel[R4r[i5],R4c[i5]]
        i5=i5+1
    
    #rule three and four
    R6r, R6c= np.where((Ydash>Cr) & (Cb> Cr) & (Cr<(7.4*Crstd)))
    ruleVIpixel=len(R6r)
    Ir6= np.zeros(np.shape(img1), dtype='uint8')
    for i6 in range(ruleVIpixel):
        Ir6[R6r[i6],R6c[i6],2] =rchannel[R6r[i6],R6c[i6]]
        Ir6[R6r[i6],R6c[i6],1] =gchannel[R6r[i6],R6c[i6]]
        Ir6[R6r[i6],R6c[i6],0] =bchannel[R6r[i6],R6c[i6]]
        i6=i6+1
    
    f_f = cv2.add(Ir12,Ir6)


    R7r, R7c = np.where( (f_f[:,:,1] > f_f[:,:,0] ) & ( f_f[:,:,1] > f_f[:,:,0]) & (f_f[:,:,0 ]< 100))
    rows, columns = np.shape(img1[:,:,2])
    #columns = len(img1[:,:,0])
    blank=f_f
    blank[np.where((blank == [255,255,255]).all(axis = 2) | (blank != [0,0,0]).all(axis = 2) )] = [128,0,128]

    ff_c= 1
    ff_r=len(R7r)
    
    if ff_r > ( rows*columns*2/100):
       fdet=cv2.putText(frame,"Fire detected ", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,lineType=cv2.LINE_AA)
       fdet=cv2.add(fdet,blank)
       fdet=cv2.resize(fdet, (960, 520))
       cv2.imshow("Fire Detection",fdet)
       blank=cv2.putText(blank,"Fire detected ", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),1, lineType=cv2.LINE_AA)
       blank=cv2.resize(blank, (960, 520))
       cv2.imshow("Firepixels",blank)
       fire=True
           
    else:
         Ifinal= np.zeros(np.shape(img1), dtype='uint8')
         fnd=cv2.putText(frame,"Fire NOT detected ", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,lineType=cv2.LINE_AA)
         fnd=cv2.resize(fnd, (960, 520))
         cv2.imshow("Fire Detection",fnd)
         fire=False
    # fire detecton code ends here
    
    #List to store the human detections
    if fire==True:
        
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
            #fd = hog(im_window, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
                 fd=hog(im_window, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(2, 2),block_norm='L1', visualise=False, normalise=None)
                 fd = fd.reshape(1, -1)
                 pred = clf.predict(fd)
     
                 if pred == 1:
                
                    if clf.decision_function(fd) > 0.5:
                        detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), clf.decision_function(fd), 
                        int(min_wdw_sz[0] * (downscale**scale)),
                        int(min_wdw_sz[1] * (downscale**scale))))
                 

            
             scale += 1

        clone = im.copy()

        for (x_tl, y_tl, _, w, h) in detections:
            cv2.rectangle(im, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 255, 0), thickness = 2)

        rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
        sc = [score[0] for (x, y, score, w, h) in detections]
        print ("sc: ", sc)
        sc = np.array(sc)
        pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.3)
        if sc.any() < 0.7:
           clone=cv2.putText(clone,"Human not Detected", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255,255),1, lineType=cv2.LINE_AA)
           im=cv2.putText(im,"Human not Detected", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),1.5, lineType=cv2.LINE_AA)
        else:
            clone=cv2.putText(clone,"Human Detected", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),1, lineType=cv2.LINE_AA)
            im=cv2.putText(im,"Human Detected", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),1, lineType=cv2.LINE_AA)
    #print ("shape, ", pick.shape)

        for(xA, yA, xB, yB) in pick:
            cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 255, 0), 2)
    
        cv2.imshow('before NMS',im)
        cv2.imshow('after NMS',clone)
    
    
    
    key=cv2.waitKey(1)
    if key == ord('q'):
       break
cap.release()
cv2.destroyAllWindows()
    




