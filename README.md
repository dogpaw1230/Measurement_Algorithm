# 🌱 프로젝트 소개

### 기술 스택
**Python, OpenCV, Pandas, Numpy**

### 1. 연구 제목
> OpenCV 기반의 머신 비전 알고리즘을 사용한 재료 변형 측정 기술 개발

### 2. 연구 배경

<img width:700px src="https://github.com/dogpaw1230/Measurement_Algorithm/assets/146051611/49b3bca1-e860-4450-aea4-54947fd936b5" alt="연구배경">

<br>

  1. 굽힘은 재료의 모양을 변화시키는 기본 메커니즘으로 자동차, 선박, 플랜트 등 다양한 산업에 사용됨
  2. 특히 비강도가 높은 **마그네슘 합금**은 반도체, 디스플레이, 자동차와 같은 중량 저하가 요구되는 다양한 분야에 사용되고 있음
  3. 마그네슘 합금은 **인장 및 압축에서 비대칭 거동**을 갖기 때문에 유한요소해석으로는 굽힘 변형을 분석하기 여려움
  4. 특히 재료 곡률과 같은 **형상 변형을 실시간으로 분석할 수 있는 연구 및 방법이 부족한 실정**임



### 3. 연구 목적

  1. 스마트폰 카메라와 Python 기반의 OpenCV를 활용한 영상 처리를 통해 재료가 굽힘 변형할 때, 성형 곡률을 측정하는 방법 제시
  2. 측정한 성형 곡률과 재료 시험기의 하중 데이터를 이용하여 곡률-모멘트 선도를 도출하고, 마그네슘 합금의 항복점과 탄성 계수를 도출하여 재료 성질 분석

- V 굽힘 실험 구성 및 재료 변형 측정 예시
  
<img width:700px src="https://github.com/dogpaw1230/Measurement_Algorithm/assets/146051611/4d73979d-2901-4dbe-817f-ae1d16c9139a" alt="실험장비_측정예시">

<br><br>


# ✨ 프로젝트 내용

### 1. 카메라 캘리브레이션

<details>
  <summary>캘리브레이션 설명 (펼쳐보기)</summary>
    <img width:700px src="https://github.com/dogpaw1230/Measurement_Algorithm/assets/146051611/d7a1b930-0130-4d22-a7d0-09c76e3a072f" alt="카메라캘리브레이션">
    <img width:700px src="https://github.com/dogpaw1230/Measurement_Algorithm/assets/146051611/c9d903ad-9b4a-470b-be74-4f781d57d324" alt="카메라캘리브레이션">
    <img width:700px src="https://github.com/dogpaw1230/Measurement_Algorithm/assets/146051611/e844472b-8ba2-4e5d-8493-27b783b5ed49" alt="카메라캘리브레이션">
</details>


  1. 렌즈 왜곡 계수는 [다크프로그래머](https://darkpgmr.tistory.com/32) 블로그에서 제공하는 tool을 사용하여 도출함
  2. Calibration 폴더 안에 있는 **4개의 코드로 실험 영상을 보정할 수 있음** 
     - 실험 영상의 초당 프레임(fps)이 60 이기 때문에 컴퓨터 메모리의 문제로 한번에 보정이 불가 <Br>
       ➔ 실험 영상을 초당 1장의 이미지로 저장 (ex. 실험 영상이 2분이면 총 120장의 이미지로 저장) **Calibration/1_img_write.py** 코드를 참고
     - 렌즈 왜곡 계수를 사용해 렌즈 왜곡으로 발생하는 이미지 왜곡을 보정 **Calibration/2_img_calibauto.py** 코드를 참고
     - 기준점을 설정하여 이미지를 원근 변환 **Calibration/3_img_perspective.py** 코드를 참고
     - 보정이 완료된 이미지를 다시 동영상으로 변환 **Calibration/4_img_to_video.py** 코드를 참고
    
<br>

### 2. HSV 색변환을 이용한 외곽선 좌표 검출

<details>
  <summary>HSV 색변환 및 외곽선 좌표 검출 설명 (펼쳐보기)</summary>
    <img width:700px src="https://github.com/dogpaw1230/Measurement_Algorithm/assets/146051611/4ca0aeb8-c0b2-4bea-a691-a3fc98d9345c" alt="외곽선검출">
</details>


  1. 실험 영상에 HSV 색변환을 적용한 뒤, 재료의 외곽선 좌표를 검출하여 csv 파일로 저장
  2. 검출한 좌표 데이터를 가공하여 재료의 성형 곡률 및 각도를 계산
  3. **Curvature and Angle Measurement/Curvature and angle measurement p to m.py** 코드를 참고

<br>

### 3. 재료의 성형 곡률 측정

<details>
  <summary>성형 곡률 측정 방법 - Fixed Span 방법 (펼쳐보기)</summary>
    <img width:700px src="https://github.com/dogpaw1230/Measurement_Algorithm/assets/146051611/0cad57c8-1c40-4cfd-9fee-c136b5f04ff1" alt="FixedSpan">
</details>

  1. Pandas와 Numpy 라이브러리를 활용하여 재료의 굽힘 변형이 일어나는 영역의 좌표 데이터를 필터링
  2. 필터링한 좌표 데이터에서 y값이 가장 큰 값을 찾고, 그 점을 재료 변형부의 중심으로 가정
  3. 중심점의 좌우로 곡률을 측정하기 위한 적정한 데이터 길이(Span)를 설정 ➔ 본 연구에서는 400 픽셀로 설정
  4. 측정 오차는 Least Square Method의 residual로 정함
  5. **Curvature and Angle Measurement/Curvature and angle measurement p to m.py** 코드를 참고

<br>



