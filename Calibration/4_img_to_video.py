import cv2
import numpy as np
import glob, natsort
import datetime
import time
 
j = 1
now = time.localtime()
s = "%04d-%02d-%02d" % (now.tm_year, now.tm_mon, now.tm_mday)
experiname = '/Vbending/'

for i in range(1, 3):
    path = s+experiname+'test'+ str(i) +'/calibrated img/*jpg'
    images = glob.glob(path)
    img_list = natsort.natsorted(images)
    print(img_list)

    img_array = []
    for name in img_list:
        print(name)
        img = cv2.imread(name)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
     
    path1 = s+experiname+'test' + str(i) +'/img to video/'
    out = cv2.VideoWriter(path1+str(i)+'_Vbending.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
     
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    
    j += 1