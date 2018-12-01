# -*- coding: utf-8 -*-
"""
Created on Tue Oct 7 11:42:45 2018
@author: Animesh
"""
import cv2 as cv
#import numpy as np
import os

#img = cv2.imread("pos_2.jpg", 0) #Read Image as Numpy Array
mytemplate = cv.imread('template_animesh.png', 0)
laplacian_template = cv.Laplacian(mytemplate, cv.CV_8U)
w, h = laplacian_template.shape[::-1]

techniques = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 
           'cv.TM_CCORR_NORMED','cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

#Stores all 6 techniques of comparison in a list

source_images = ['neg_1.jpg','neg_2.jpg','neg_3.jpg','neg_4.jpg','neg_5.jpg','neg_6.jpg','neg_8.jpg','neg_9.jpg','neg_10.jpg','pos_1.jpg','pos_2.jpg','pos_3.jpg','pos_4.jpg','pos_5.jpg','pos_6.jpg','pos_7.jpg','pos_8.jpg','pos_9.jpg','pos_10.jpg','pos_11.jpg','pos_12.jpg','pos_13.jpg','pos_14.jpg','pos_15.jpg']

for tech in techniques:
    if not os.path.exists(tech):
        os.makedirs('./outputs/' + tech)
    
    for image in source_images:
        img = cv.imread(image, 0)
        img_blur = cv.GaussianBlur(img, (3, 3), 0)
        img_blur_laplacian = cv.Laplacian(img_blur, cv.CV_8U)

        method = eval(tech)

        # Apply template Matching
        # res = cv.matchTemplate(img_blur_laplacian, laplacian_template, method)
        res = cv.matchTemplate(img, mytemplate, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv.rectangle(img, top_left, bottom_right, 255, 2)
        cv.imwrite('./outputs/' + tech + '/' + image, img)

