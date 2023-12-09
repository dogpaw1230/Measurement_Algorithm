import os
import glob
import natsort
import cv2
import datetime
import time

# 날짜 및 시간으로 폴더 만들기
now = time.localtime()
s = "%04d-%02d-%02d" % (now.tm_year, now.tm_mon, now.tm_mday)
os.makedirs(s, exist_ok=True)

# video 폴더 안 실험 영상 읽어오기
file = glob.glob('./Video1/*.MOV')
video_list = natsort.natsorted(file)
#print(video_list)

for i in video_list:
    # 숫자 네이밍용 코드
    files = i.split('Video1/')[1] # 파일 이름 ex) Tensile_test1.MOV
    files0 = files[:].split('_')[0] # Tensile
    files1 = files[:].split('_')[1] # test1.MOV
    files2 = files1[:].split('.M')[0] # test1

    path = s+'/'+files0
    os.makedirs(path, exist_ok=True) # Tensile
    os.mkdir(s+'/'+files0+'/'+files2) # Tensile>test1
    os.mkdir(s+'/'+files0+'/'+files2+'/'+'original img')
    os.mkdir(s+'/'+files0+'/'+files2+'/'+'perspective transformed img')
    os.mkdir(s+'/'+files0+'/'+files2+'/'+'calibrated img')
    os.mkdir(s+'/'+files0+'/'+files2+'/'+'img to video')

    # 실험 영상 읽어오기 및 좌표 추출
    cap = cv2.VideoCapture('./Video1/'+files)
    count = 0

    while(cap.isOpened()):
        ret, image = cap.read()
        if ret:
            if(int(cap.get(1)) % 1 == 0): #앞서 불러온 fps 값을 사용하여 1초마다 추출
                cv2.imwrite(s+'/'+files0+'/'+files2+"/original img/frame"+str(int(cap.get(1)))+'.jpg', image)
                print('Saved frame number :', str(int(cap.get(1))))
                count += 1
        else:
            print("[프레임 수신 불가] - 종료합니다")
            break   
    
    cap.release()
