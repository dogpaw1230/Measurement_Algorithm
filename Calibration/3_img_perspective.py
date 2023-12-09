import cv2, glob, natsort
import numpy as np
import datetime
import time

now = time.localtime()
s = "%04d-%02d-%02d" % (now.tm_year, now.tm_mon, now.tm_mday)
images = glob.glob(s+'/Vbending/test2/original img/*.jpg')
img_list = natsort.natsorted(images)
print(img_list)

for name in img_list:
	print(name)
	files = name.split('img/')[1] # ex) frame0.jpg
	img = cv2.imread(name)

	# 원근 변환 전 4개 좌표
	tl = [2*148, 2*332]
	bl = [2*162, 2*923]
	tr = [2*1642, 2*337]
	br = [2*1624,2*921]
	pts1 = np.float32([tl, tr, br, bl])

	# 변환 후 영상에 사용할 점 사이의 폭과 높이 계산
	w1 = abs(br[0] - bl[0])    # 상단 좌우 좌표간의 거리
	w2 = abs(tr[0] - tl[0])          # 하단 좌우 좌표간의 거리
	h1 = abs(tr[1] - br[1])      # 우측 상하 좌표간의 거리
	h2 = abs(tl[1] - bl[1])        # 좌측 상하 좌표간의 거리
	width = max([w1, w2])                       # 두 좌우 거리간의 최대값이 점 사이의 폭
	height = max([h1, h2])                      # 두 상하 거리간의 최대값이 점 사이의 높이

	# w1 = 2924
	# w2 = 2988
	# h1 = 1168
	# h2 = 1182


	# 변환 후 4개 좌표
	pts2 = np.float32([[0,0], [width-1,0], [width-1,height-1], [0,height-1]])
	# pts2 = np.float32([[0,0], [w2-1,0], [w2-1,h2-1], [0,h2-1]])

	# 변환 행렬 계산 
	mtrx = cv2.getPerspectiveTransform(pts1, pts2)
	# 원근 변환 적용
	result = cv2.warpPerspective(img, mtrx, (int(width), int(height)))
	# cv2.imshow('perspective', result)
	cv2.imwrite(s+'/Vbending/test2/perspective transformed img/'+files,result)

	# cv2.imshow("origin", img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()