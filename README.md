# 😁 HAPPY 😁: HAte sPeech Purification for You 

TODO: ADD SIMPLE DEMO PICTURE
![image](https://user-images.githubusercontent.com/68216991/217414599-1a4bf439-3df6-42de-b86a-3a92487d267f.png)

- 댓글을 작성하면, 분류 모델이 혐오 표현인지를 먼저 판별합니다.
- 혐오 표현으로 분류될 경우, 토큰 분류 모델을 이용해 문장의 어느 부분이 혐오 표현인지를 찾아 알려줍니다.
- 마지막으로 생성 모델을 이용해, 문장의 순화된 내용을 생성하여 사용자에게 순화 방향을 제시해 줍니다.

## 악성 댓글 분류 및 순화 재생성 프로젝트
- 딥러닝을 이용해 댓글의 혐오 여부를 분류하고, 혐오표현으로 판단된 경우 의미를 유지한 문장을 재생성합니다.  
- 이 과정을 통해 사용자의 문제의식을 일으키고 자발적 개선을 유도합니다 

## 목차  
1. [팀원 소개](팀원-소개)
2. [서비스 ARCHITECTURE](서비스-ARCHITECTURE)
3. [모델 구조](모델-구조)
4. [데이터](데이터)
5. [추가 정보](추가-정보)

## 팀원 소개
| [김준휘](https://github.com/intrandom5) | [류재환](https://github.com/risolate) | [박수현](https://github.com/HitHereX) | [박승현](https://github.com/koohack) | [설유민](https://github.com/ymnseol) |
|--|--|--|--|--|
|![image](https://user-images.githubusercontent.com/112468961/217268409-ef9dc7fa-de0d-41e1-9bd2-bb6822bd57fd.png)|![image](https://user-images.githubusercontent.com/112468961/217268472-86fdf5c2-268e-488e-adbb-5a5b3258f905.png)|![image](https://user-images.githubusercontent.com/112468961/217268511-64874022-8951-4d69-88d1-f07bfe16045e.png)|![image](https://user-images.githubusercontent.com/112468961/217268552-ac1a368d-90da-4c5a-a9f2-4c9f22345bea.png)|![image](https://user-images.githubusercontent.com/112468961/217268600-aa9a5e41-0dad-43bc-afe2-730326800fc4.png)|
|Classification model</br> Classification API</br> Data Collecting</br>|Generation Model</br> Generation API</br> Data Collecting|Classification Model</br> Data Guideline</br> Data Collecting</br> Data Checking|Generation Model</br> Database</br> BackEnd</br> FrontEnd</br> Data Web</br> Data Collecting|Generation Model</br> Data Collecting</br> Data Checking|

## 서비스 ARCHITECTURE
![image](https://user-images.githubusercontent.com/112468961/217270747-76faf62a-b9ca-4599-98fc-8fd790bde451.png)
TOTALK : 말로 풀어서 설명이필요할까...?(나는 아닌 것 같음)  
TODO : SCENARIO TEST PIC

## 모델 구조

### CLASSIFICATION MODEL
![image](https://user-images.githubusercontent.com/112468961/217277163-50f58a95-21f0-49de-89ad-550d76d68da1.png)
- base 모델로는 가장 높은 F1 score를 보이면서도 합리적인 추론 시간을 보인 KcElectra-base 모델을 사용했습니다
  - F1 score 90.88 TODO : micro vs macro
  - RPS : 173
### GENERATION MODEL
TODO : ADD final selection after human evaluation

## 데이터
### CLASSIFICATION MODEL
- 혐오 문장 분류 모델의 학습에는 한국어 뉴스기사 댓글에서 수집한 혐오표현 데이터셋인 [K-MHaS](https://github.com/adlnlp/K-MHaS) 를 사용했습니다.
- 혐오표현 토큰 분류 모델의 학습에는 네이버 뉴스와 유튜브 영상 댓글에서 수집한 한국어 혐오표현 데이터셋인 [KOLD](https://github.com/boychaboy/KOLD) 를 사용했습니다

### GENERATION MODEL : Parallel Dataset 제작
- 혐오표현을 제거하되 의미를 유지한 문장 재생성 학습을 위해, 직접 혐오표현-순화표현 parallel dataset(총 xxxx개) 을 구축했습니다.
- 구축한 dataset에 대한 보다 자세한 정보는 /data 폴더의 readme 에서 확인하실 수 있습니다

## 추가 정보
TODO:links
- 발표영상 링크
- 발표자료 링크(GITHUB에 PDF로 추가)
- 랩업리포트


