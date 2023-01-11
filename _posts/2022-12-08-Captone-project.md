---
layout: post
title: Mediapipe
author: [李昱 陳毅展 張雅婷]
category: [Lecture]
tags: [jekyll, ai]
---

期末專題

## Mediapipe姿態辨識
MediaPipe 是 Google Research 所開發的多媒體機器學習模型應用框架，支援 JavaScript、Python、C++ 等程式語言，可以運行在嵌入式平臺 ( 例如樹莓派等 )、移動設備 ( iOS 或 Android ) 或後端伺服器，目前如 YouTube、Google Lens、Google Home 和 Nest...等，都已和 MediaPipe 深度整合。<br>
Mediapipe Pose 模型可以標記出身體共 33 個姿勢節點的位置，甚至可以進一步透過這些節點，將人物與背景分離，做到去背的效果，下圖標示出每個節點的順序和位置 <br>

### 如果使用 Python 語言進行開發，MediaPipe 支援下列幾種辨識功能：
* **MediaPipe Face Detection ( 人臉追蹤 )**<br>
* **MediaPipe Face Mesh ( 人臉網格 )**<br>
* **MediaPipe Hands ( 手掌偵測 )**<br>
* **MediaPipe Holistic ( 全身偵測 )**<br>
* **MediaPipe Pose ( 姿勢偵測 )**<br>
* **MediaPipe Objectron ( 物體偵測 )**<br>
* **MediaPipe Selfie Segmentation ( 人物去背 )**<br>
---
---

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
**基本設置與角度計算函數**
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
ExAngle=90
ExStatus=False
ExStatus1=False
ExStatus2=False
ExStatus3=False
countEx=0
countEx1=0
countEx2=0
countEx3=0
a=0
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
  if HandAngel<90 and ExStatus==False :
    countEx=1
    ExStatus=True
  elif HandAngel>90 :
    countEx=0
    ExStatus=False
  else:
    countEx=0
  return countEx
  
  
def countExF1(LegAngel): 
  global ExStatus1
  if(LegAngel<90 and Angle>120 ) and ExStatus1==False :
    countEx1=1
    ExStatus1=True
  elif LegAngel>90 and Angle>120:
    countEx1=0
    ExStatus1=False
  else:
    countEx1=0
  return countEx1


def countExF2(ArmpitAngel): 
  global ExStatus2
  if ArmpitAngel<90 and ExStatus2==False:
    countEx2=1
    ExStatus2=True
  elif ArmpitAngel>90 :
    countEx2=0
    ExStatus2=False
  else:
    countEx2=0
  return countEx2

def countExF3(Angel): 
  global ExStatus3
  if ((Angel<90 and LegAngle<90 )and ExStatus3==False)  :
    countEx3=1
    ExStatus3=True
  elif Angel>90 or LegAngle>90 :
    countEx3=0
    ExStatus3=False
  else:
    countEx3=0
  return countEx3
```
**定義所需的mediapipe身體標點**
``` 
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
      #左腳軸3點->23、25、27
      if (not results.pose_landmarks==None): #至少有一個身體
        a=np.array([results.pose_landmarks.landmark[23].x*imgW,results.pose_landmarks.landmark[23].y*imgH])
        b=np.array([results.pose_landmarks.landmark[25].x*imgW,results.pose_landmarks.landmark[25].y*imgH])
        c=np.array([results.pose_landmarks.landmark[27].x*imgW,results.pose_landmarks.landmark[27].y*imgH])

        LegAngle=FindAngleF(a,b,c)

        d=np.array([results.pose_landmarks.landmark[11].x*imgW,results.pose_landmarks.landmark[11].y*imgH])
        e=np.array([results.pose_landmarks.landmark[13].x*imgW,results.pose_landmarks.landmark[13].y*imgH])
        f=np.array([results.pose_landmarks.landmark[15].x*imgW,results.pose_landmarks.landmark[15].y*imgH])

        HandAngle=FindAngleF(d,e,f) #算出角度

        g=np.array([results.pose_landmarks.landmark[14].x*imgW,results.pose_landmarks.landmark[14].y*imgH])
        h=np.array([results.pose_landmarks.landmark[12].x*imgW,results.pose_landmarks.landmark[12].y*imgH])
        i=np.array([results.pose_landmarks.landmark[24].x*imgW,results.pose_landmarks.landmark[24].y*imgH])

        ArmpitAngle=FindAngleF(g,h,i) #算出角度

        j=np.array([results.pose_landmarks.landmark[24].x*imgW,results.pose_landmarks.landmark[24].y*imgH])
        k=np.array([results.pose_landmarks.landmark[26].x*imgW,results.pose_landmarks.landmark[26].y*imgH])
        l=np.array([results.pose_landmarks.landmark[28].x*imgW,results.pose_landmarks.landmark[28].y*imgH])

        Angle=FindAngleF(j,k,l) #算出角度
    ```
    **計算動作執行次數**
    ```
        #算出次數
        
        
        countEx=countEx+countExF(HandAngle)
        print("countEx=",countEx)
    

        countEx1=countEx1+countExF1(LegAngle)
        print("countEx1=",countEx1)

        countEx2=countEx2+countExF2(ArmpitAngle)
        print("countEx2=",countEx2)

        countEx3=countEx3+countExF3(Angle)
        print("countEx3=",countEx3)

        if countEx > 10:
            countEx=10
       ```
       **定義和顯示出介面**
       ```
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
           cv2.putText(image, str(round(LegAngle,2)) , (x25,y25), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
           cv2.putText(image,  str("kick:") , (20,90), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
           cv2.putText(image,  str(countEx1) , (150,90), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        if(countEx1>9):
            cv2.putText(image,  str("(finish)") , (220,90), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
        x13,y13=round((1-results.pose_landmarks.landmark[13].x)*imgW),int(results.pose_landmarks.landmark[13].y*imgH)
        if (x13<imgW and x13>0) and (y13<imgH and y13>0):
           cv2.putText(image, str(round(HandAngle,2)) , (x13,y13), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)  
           cv2.putText(image,  str(countEx) , (260,40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
           cv2.putText(image,  str("hand up:") , (20,40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        

        if(countEx>9):
           cv2.putText(image,  str("(finish)") , (330,40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        

        x26,y26=round((1-results.pose_landmarks.landmark[26].x)*imgW),int(results.pose_landmarks.landmark[26].y*imgH)
        if (x26<imgW and x26>0) and (y26<imgH and y26>0):
           cv2.putText(image, str(round(HandAngle,2)) , (x26,y26), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3) 
           cv2.putText(image,  str("squat:") , (20,140), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
           cv2.putText(image,  str(countEx3) , (180,140), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        if(countEx3>9):
           cv2.putText(image,  str("(finish)") , (250,140), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        

        x12,y12=round((1-results.pose_landmarks.landmark[12].x)*imgW),int(results.pose_landmarks.landmark[12].y*imgH)
        if (x12<imgW and x12>0) and (y12<imgH and y12>0):
          cv2.putText(image, str(round(ArmpitAngle,2)) , (x12,y12), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
          cv2.putText(image,  str("hand down:") , (20,190), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
          cv2.putText(image,  str(countEx2) , (310,190), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        if(countEx2>9):
           cv2.putText(image,  str("(finish)") , (380,190), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        

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
     
      if(countEx>9 and countEx1>9 and countEx2>9 and countEx3>9):
         print("good job")
         break

cap.release()
```
### 測試結果
![](https://github.com/JULIA1021/AI-course/blob/gh-pages/images/1.jpg?raw==true)

<iframe width="560" height="315" src="https://www.youtube.com/embed/TTE3SjYuing" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>


<br />
<br />

This site was last updated {{ site.time | date: "%B %d, %Y" }}.
