import os
import glob
import natsort
import math
import sympy
import datetime
import time
import cv2
import numpy as np
import pandas as pd
import circle_fit as cf
import matplotlib.pyplot as plt


def make_circle(c, r):
    theta = np.linspace(0, 2 * np.pi, 256)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.vstack((x, y)).T + c

# 날짜 및 시간으로 폴더 만들기
now = time.localtime()
s = "%04d-%02d-%02d" % (now.tm_year, now.tm_mon, now.tm_mday)
os.mkdir(s) 

# video 폴더 안 실험 영상 읽어오기
file = glob.glob('./Video_Data/*.MOV')
video_list = natsort.natsorted(file)
#print(video_list)

for i in video_list:
    # 숫자 네이밍용 코드
    files = i.split('optimization\\')[1]
    files0 = files[:].split('_')[0] # R63t
    files1 = files[:].split('_')[1]
    files2 = files1[:].split('.M')[0] # Stroke 18mm

    # 좌표 데이터와 성형 각도 데이터 저장 폴더 만들기
#    if os.path.exists(s+'/'+files0)
#        continue
#    os.mkdir(s+'/'+files0)
    path = s+'/'+files0
    os.makedirs(path, exist_ok=True)
    os.mkdir(s+'/'+files0+'/'+files2)
    os.mkdir(s+'/'+files0+'/'+files2+'/'+'coordi data')
    os.mkdir(s+'/'+files0+'/'+files2+'/'+'angle')
    os.mkdir(s+'/'+files0+'/'+files2+'/'+'curvature')
    os.mkdir(s+'/'+files0+'/'+files2+'/'+'point data for curvature')
    os.mkdir(s+'/'+files0+'/'+files2+'/'+'data for fitting')
    os.mkdir(s+'/'+files0+'/'+files2+'/'+'coordi data p to mm')

    # 실험 영상 읽어오기 및 좌표 추출
    capture = cv2.VideoCapture('./Video Data/video for stroke optimization/'+files)
    count = 0

    # 30프레임 마다 데이터 저장
    savecoordinatefreq = 10


    while capture.isOpened():

        run, img_color = capture.read()
        img_color = cv2.flip(img_color, 0)
        # h, w = img_color.shape[:2]
#        img_color = cv2.resize(img_color, None, None, 0.4, 0.4,  interpolation=cv2.INTER_AREA)

        if not run:
            print("[프레임 수신 불가] - 종료합니다")
            break

        kernel = np.ones((5, 5), np.uint8)


    # hsv이미지로 변환한다.
        img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
        lower_red = (165, 100, 100)
        upper_red = (180, 255, 255)
        lower_red1 = (0, 100, 100)
        upper_red1 = (15, 255, 255)
        mask_red = cv2.inRange(img_hsv, lower_red, upper_red)
        mask_red1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
#        img_mask = mask_red | mask_red1
        img_mask = mask_red 

        dilation = cv2.dilate(img_mask, kernel, iterations=5)
        erosion = cv2.erode(dilation, kernel, iterations=5)
        # cv2.imshow("result2", img_hsv)
        # cv2.imshow("result2", erosion)
        # cv2.imshow("result2", img_mask)

        # findcontours 함수는 윈도우 환경에서 인자를 3개 반환한다 -> image, contours, hierarchy
        im, contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            img_color = cv2.drawContours(img_color, contours, -1, (0, 255, 0), 2)
        
        text = s+'/'+files0+'/'+files2+'/coordi data/'+str(int(capture.get(1)))
        text += ".csv"
        print(contour)
        countpoint = 0
        # contours = 컨투어의 개수, i = 컨투어 좌표 리스트 안의 (x,y), j = j[0]은 x좌표/ j[1]은 y좌표
        for k in contours:
            if (count % savecoordinatefreq) == 0:
                f = open(text, 'w')
                for i in k:
                    for j in i:
                        print(j[0], ',', j[1], file=f)

                    # countpoint += 1
                    # if countpoint > 1480:
                    #     print('Done!')
                    #     break

            count += 1
        f.close()
        cv2.imshow("result", img_color)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    capture.release()
    cv2.destroyAllWindows()
    
    bfile = glob.glob(s+'/'+files0+'/'+files2+'/coordi data/*.csv')
    bcsv_list = natsort.natsorted(bfile)
    for p in bcsv_list:
        bfiles = p.split('data\\')[1]
        bfiles1 = bfiles[:].split('.')[0]
        
        bdf = pd.read_csv(p, header=None)
        xcm = bdf[:]*0.027
        xcm.to_csv(s+'/'+files0+'/'+files2+'/'+'coordi data p to mm/'+str(bfiles1)+'.csv', index=False, header=None)
        

    name = ['Frame num', 'xc', 'yc', 'r', 'resi', 'curvature']
    # 곡률 및 성형 각도 구하기
    afile = glob.glob(s+'/'+files0+'/'+files2+'/coordi data p to mm/*.csv')
    csv_list = natsort.natsorted(afile)
    print(csv_list)
    angle = []
    curvature = []
    for j in csv_list:
        # 숫자 네이밍 용 코드
        afiles = j.split('mm\\')[1]
        afiles1 = afiles[:].split('.')[0]
        
        df = pd.read_csv(j, header=None)
        x_max = df[0].max()  # 첫번째 열에서 가장 큰 값을 가져온다.
        x_min = df[0].min()  # 첫번째 열에서 가장 작은 값을 가져온다. 
        xmax_index = df[0].idxmax()  # 첫번째 열에서 가장 큰 값의 인덱스를 가져온다.
        xmin_index = df[0].idxmin()  # 첫번째 열에서 가장 작은 값의 인덱스를 가져온다.
        
        # 첫번째 열에서 가장 큰 값, 즉 가장 큰 x 픽셀 값 주변 20퍼센트에 해당하는 x 값들의 인덱스를 가져온다.
        req_xmax = df[(df[0] >= 0.95 * x_max) & (df[0] <= 1.05 * x_max)].index.tolist()
        data_xmax = df.loc[req_xmax]  # req_xmax에 해당하는 데이터를 추출해서 data에 저장
        ymax_index = data_xmax[1].idxmax()  # data의 y값 중 최대값의 인덱스를 찾아 ymax_index에 저장
        end_index = df.index[ymax_index]-50
        
        # 첫번째 열에서 가장 작은 값, 즉 가장 작은 x 픽셀 값 주변 20퍼센트에 해당하는 x 값들의 인덱스를 가져온다. 
        req_xmin = df[(df[0] >= 0.95 * x_min) & (df[0] <= 1.05 * x_min)].index.tolist()
        data_xmin = df.loc[req_xmin]  # req_index1에 해당하는 데이터를 추출해서 data에 저장
        ymin_index = data_xmin[1].idxmin()  # data의 y값 중 최소값의 인덱스를 찾아 ymin_index에 저장
        start_index = df.index[ymin_index]+10
        
        data_for_fitting = df.loc[start_index:end_index]  # 전체 데이터 프레임에서 ymin 인덱스에 해당하는 데이터까지만 추출해서 data1에 저장
#        skipnum = len(data1) * 0.05  # 데이터 앞,뒤로 5%씩 인덱스 계산
#        start = skipnum
#        end = ymin_index - skipnum
#        data_for_fitting = df.loc[start:end]  # 데이터 앞, 뒤로 5%씩 날려버리기
        # 만약 data_for_fitting에 좌표 데이터가 100개 이하면 result.csv 파일 생성하지 않기
        if len(data_for_fitting.loc[:]) < 100:
            continue
        data_for_fitting.to_csv(s+'/'+files0+'/'+files2+'/'+'data for fitting/'+str(afiles1)+'.csv')  # 그 결과값을 엑셀로 저장한다. + 가능하면 폴더 하나 생성해서!

        # 직선부 데이터 비율 정하기 및 비율에 맞게 양쪽 직선부 데이터 정렬
        scale = int(0.3 * len(data_for_fitting))
        line_data1 = data_for_fitting[:scale]
        line_data2 = data_for_fitting[-scale:]
#        line_data1.to_csv('linedata1'+str(files1)+'.csv')
#        line_data2.to_csv('linedata2'+str(files1)+'.csv')

        # 왼쪽 직선부 fitting
        linear_l = np.polyfit(line_data1[0], line_data1[1], 1)
        linear_model_l = np.poly1d(linear_l)
        # 오른쪽 직선부 fitting
        linear_r = np.polyfit(line_data2[0], line_data2[1], 1)
        linear_model_r = np.poly1d(linear_r)

        x, y = sympy.symbols('x y')
        eq1 = linear_l[0] * x - y + linear_l[1]
        eq2 = linear_r[0] * x - y + linear_r[1]
        eq3 = eq2 - eq1
        result = sympy.solve([eq1, eq2], [x, y])
        result = list(zip(result.keys(), result.values()))
        xv = result[0][1]
        yv = result[1][1]

        # x = 1500, 2100 대입해서 벡터 결정
        x1 = 1500*0.027
        x2 = 2100*0.027
        y1 = linear_model_l(x1)
        y2 = linear_model_r(x2)

        v1 = np.array([x1 - xv, y1 - yv])
        v2 = np.array([x2 - xv, y2 - yv])
        v1 = v1.astype(float)
        v2 = v2.astype(float)
        v11 = v1[0] ** 2 + v1[1] ** 2
        v22 = v2[0] ** 2 + v2[1] ** 2
        p = np.dot(v1, v2) / (np.sqrt(v11) * np.sqrt(v22))
        theta = math.acos(p)
        degree = math.degrees(theta)
        angle.append([files1, degree])
        print(degree)
        
        # 곡률 구하기
        
        # y max 점인 m 구하기
        cymax_index = data_for_fitting[1].idxmax()
        m = data_for_fitting.loc[cymax_index]

        # 점 m 양 옆으로 10%에 해당 하는 구간 추출, 총 20%임
#        b = int(len(data_for_fitting) * 0.1)
        b = 150
        arrange = data_for_fitting.loc[cymax_index - b:cymax_index + b]
        arrange.to_csv(s+'/'+files0+'/'+files2+'/point data for curvature/point data_' + str(afiles1) + '.csv')

        arrange_np = arrange.to_numpy()
        xc,yc,r,resi = cf.least_squares_circle((arrange_np))
        fitted_circle = make_circle(([xc, yc]), r)
        curvature.append([afiles1, xc, yc, r, resi, 1/r])
        print(afiles1, xc, yc, r, resi)

        
    angle_result = pd.DataFrame(angle)
    angle_result.to_csv(s+'/'+files0+'/'+files2+'/angle/Angle.csv')
    curvature_result = pd.DataFrame(curvature, columns = name)
    curvature_result.to_csv(s+'/'+files0+'/'+files2+'/curvature/Curvature.csv')
#        plt.figure(figsize=(10, 10))
#        plt.plot(line_data1[0], linear_model_l(line_data1[0]), color='green')
#        plt.plot(line_data2[0], linear_model_r(line_data2[0]), color='green')
#        plt.scatter(line_data1[0], line_data1[1], color='red')
#        plt.scatter(line_data2[0], line_data2[1], color='red')
#        plt.title('scatter plot of the whole data')
#        plt.text(1000, 1900, 'degree is %3.3f' % (degree), fontsize=20)
#        plt.xlabel('x')
#        plt.ylabel('y')
#        plt.grid()
#        plt.xlim(0, 2000)
#        plt.ylim(0, 2000)
#        plt.savefig('./degree'+str(files1)+'.png')
  
