import numpy as np
import glob, cv2
import natsort
import datetime
import time 

j = 1
now = time.localtime()
s = "%04d-%02d-%02d" % (now.tm_year, now.tm_mon, now.tm_mday)
experiname = '/Vbending/'
npzname = 'calib221120.npz'

for i in range(1, 3):
	path = s+experiname+'test'+ str(i) +'/perspective transformed img/*.jpg'
	images = glob.glob(path)
	img_list = natsort.natsorted(images)
	print(img_list)

	for name in img_list:
		print(name)
		files = name.split('img/')[1]
		img = cv2.imread(name)

		h,w = img.shape[:2]

		calib = np.load(npzname) 
		mtx = calib['mtx']
		dist = calib['dist']
		newcameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

		dst = cv2.undistort(img,mtx,dist,None,newcameraMtx)
		# dst = cv2.undistort(img,mtx,dist)

		x,y,w,h = roi

		dst = dst[y:y+h,x:x+w]

		path1 = s+experiname+'test' + str(i) +'/calibrated img/'
		cv2.imwrite(path1+files,dst)
		# cv2.imshow("result", dst)
		# cv2.waitKey(0)

	j += 1
