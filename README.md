# 😁 HAPPY 😁: HAte sPeech Purification for You 

TODO: ADD SIMPLE DEMO PICTURE
![image](https://user-images.githubusercontent.com/68216991/217414599-1a4bf439-3df6-42de-b86a-3a92487d267f.png)

- 댓글을 작성하면, 분류 모델이 혐오 표현인지를 먼저 판별합니다.
- 혐오 표현으로 분류될 경우, 토큰 분류 모델을 이용해 문장의 **어느 부분이 혐오 표현인지**를 찾아 알려줍니다.
- 생성 모델을 이용해, **문장의 순화된 내용을 생성**하여 사용자에게 순화 방향을 제시해 줍니다.

## 악성 댓글 분류 및 순화 재생성 프로젝트
- 딥러닝을 이용해 댓글의 혐오 여부를 분류하고, 혐오표현으로 판단된 경우 의미를 유지한 문장을 재생성합니다.  
- 이 과정을 통해 사용자의 문제의식을 일으키고 자발적 개선을 유도합니다.

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
![service_architecture](https://user-images.githubusercontent.com/105059564/217681238-33c787f0-4434-43f5-b94d-2a0beab37eb7.png)


## 모델 구조

### CLASSIFICATION MODEL
![classification_model_architecture](https://user-images.githubusercontent.com/105059564/217680953-813a1b02-c875-4985-bc47-3a0bb70f7311.png)

- Backbone model로는 가장 높은 F1 score를 보이면서도 합리적인 추론 시간을 보인 [🤗 beomi/KcElectra-base-v2022](https://huggingface.co/beomi/KcELECTRA-base-v2022) 모델을 사용했습니다
  - F1 score 90.88
  - RPS : 173

### GENERATION MODEL
![generation_model_architecture](https://user-images.githubusercontent.com/105059564/217681048-362aa914-608a-44ec-adb7-5677b29efe1b.png)


## 데이터
### CLASSIFICATION MODEL
- 혐오 문장 분류 모델의 학습에는 한국어 뉴스기사 댓글에서 수집한 혐오표현 데이터셋인 [K-MHaS](https://github.com/adlnlp/K-MHaS)를 사용했습니다.
- 혐오표현 토큰 분류 모델의 학습에는 네이버 뉴스와 유튜브 영상 댓글에서 수집한 한국어 혐오표현 데이터셋인 [KOLD](https://github.com/boychaboy/KOLD)를 사용했습니다.

### GENERATION MODEL : Parallel Dataset 제작
- 혐오표현을 제거하되 의미를 유지한 문장 재생성 학습을 위해, 직접 혐오 표현 - 순화 표현 parallel dataset(총 3,133개)을 구축했습니다.
- 구축한 dataset에 대한 보다 자세한 정보는 /data 폴더의 readme 에서 확인하실 수 있습니다

## 추가 정보
- 발표영상 링크
- [발표자료](https://github.com/boostcampaitech4lv23nlp1/final-project-level3-nlp-06/blob/docs-readme/%5B%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%5DNLP_6%E1%84%8C%E1%85%A9_%E1%84%8B%E1%85%A1%E1%86%A8%E1%84%91%E1%85%B3%E1%86%AF%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%85%E1%85%B2%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%20%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A1%E1%86%BC%20%E1%84%8C%E1%85%A2%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC.pdf)
- [프로젝트 소개 페이지](https://sharp-chemistry-3e6.notion.site/NLP-06-Happy-f2f3a6e1f3ec4a1aba3997ca87786300)
- 랩업 리포트
