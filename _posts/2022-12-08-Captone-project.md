---
layout: post
title: Mediapipe
author: [李昱 陳毅展 張雅婷]
category: [Lecture]
tags: [jekyll, ai]
---

期末專題

---
## Mediapipe姿態辨識

## 系統簡介及功能說明
系統簡介:角度的應用   <br>
功能說明:利用mediapipe節點偵測左腳所夾角度，來計算出角度改變的次數   <br>
![](https://github.com/JULIA1021/AI-course/blob/gh-pages/images/2.jpg?raw==true)
---
### 系統方塊圖
![](https://github.com/JULIA1021/AI-course/blob/gh-pages/images/3.jpg?raw==true)
---
### 製作步驟
1參考mediapipe 函式<br>
2在git bash使用opencv結合mediapipe<br>


**專題實作步驟**
1.開啟鏡頭<br>
2.使全身進入畫面<br>
3.將自己的左腳抬起並使膝蓋高於90度<br>

## 程式碼
```
import cv2
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_pose = mp.solutions.pose
BG_COLOR = (192, 192, 192) # gray
ExAngle=40
ExStatus=False
countEx=0
pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5)
def FindAngleF(a,b,c):    
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    if ang<0 :
      ang=ang+360
    if ang >= 360- ang:
        ang=360-ang
    return ang

def countExF(HandAngel): 
  global ExStatus
  if HandAngel<40 and ExStatus==False:
    countEx=1
    ExStatus=True
  elif HandAngel>40 :
    countEx=0
    ExStatus=False
  else:
    countEx=0
  return countEx


# For webcam input:
cap = cv2.VideoCapture(0)
with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
  with mp_pose.Pose(
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue
      imgH,imgW=image.shape[0],image.shape[1]
      #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = pose.process(image) #偵測身體
      #左手軸3點->11,13,15
      if (not results.pose_landmarks==None): #至少有一個身體
        a=np.array([results.pose_landmarks.landmark[23].x*imgW,results.pose_landmarks.landmark[11].y*imgH])
        b=np.array([results.pose_landmarks.landmark[25].x*imgW,results.pose_landmarks.landmark[13].y*imgH])
        c=np.array([results.pose_landmarks.landmark[27].x*imgW,results.pose_landmarks.landmark[15].y*imgH])
        #算出角度
        HandAngle=FindAngleF(a,b,c)
        #print(HandAngle)
        #算出次數

        countEx=countEx+countExF(HandAngle)
        print("countEx=",countEx)
        #畫出點位
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks, #點
            mp_pose.POSE_CONNECTIONS, #連線
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
          )
        image=cv2.flip(image, 1)
        x25,y25=round((1-results.pose_landmarks.landmark[25].x)*imgW),int(results.pose_landmarks.landmark[25].y*imgH)
        if (x25<imgW and x25>0) and (y25<imgH and y25>0):
          cv2.putText(image, str(round(HandAngle,2)) , (x25,y25), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
        cv2.putText(image,  str(countEx) , (30,40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        # 畫面切割
        # image.flags.writeable = True
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        # bg_image = np.zeros(image.shape, dtype=np.uint8)
        # bg_image[:] = BG_COLOR
        # image = np.where(condition, image, bg_image)
        # image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #顯示結果      
      cv2.imshow('MediaPipe Pose',image)
      if cv2.waitKey(5) & 0xFF == 27:
        break
      if countEx==20:
        printf("you got it")
        break
        
cap.release()
```
### 測試結果
![](https://github.com/JULIA1021/AI-course/blob/gh-pages/images/1.jpg?raw==true)

<iframe width="560" height="315" src="https://youtube.com/embed/NbBt8I_7Q9k" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>


<br />
<br />

This site was last updated {{ site.time | date: "%B %d, %Y" }}.
