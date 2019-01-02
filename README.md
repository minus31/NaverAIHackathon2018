# 네이버 AI 해커톤 2018_Ai Vision
-------------
## TEAM : GodGam (Hyun, Jaeyoung Yoon, Dooyoung Ryu)

## Envirionment

  - Python 3.6

## Library
  - TensorFlow 1.11
  - torch 0.4
  - keras 2.2

## files
  - `setup.py` : NSML 세션의 환경 설정
  - `requirements.txt` : 의존 패키지 리스트 (pip로 설정 가능한 것)
  - `main.py` : 기본 entry 파일
  - `.nsmlignore` : Session에 전달하지 않을 파일 목록

## Directories
  - `baseline` : Naver에서 제공한 베이스라인 모델 (mAP : 0.0116)

## Training data  : 상품 이미지
  - number of classes : 1000
  - total volume : 7,104
  - 이미지는 저작권의 이유로 비공개, 다만 하나의 클래스의 이미지를 공개 했음.

  <p float="left">
    <img src="./sample_image/ex (1).jpg" width="100" />
    <img src="./sample_image/ex (2).jpg" width="100" />
    <img src="./sample_image/ex (3).jpg" width="100" />
    <img src="./sample_image/ex (4).jpg" width="100" />
    <img src="./sample_image/ex (5).jpg" width="100" />
    <img src="./sample_image/ex (6).jpg" width="100" />
  </p>

## Test data
  - Query Image를 질의 하면, Reference image 에서 질의한 이미지와 같은 이미지를 결과로 출력한다.
  - 1,322 장의 이미지 중, Query 이미지 : 195, Reference 이미지 : 1,127

  Metric
  - [mAP(mean average precision)](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)
  - 동점자의 경우 Recall@K 계산

  ## Baseline Model
  - Keras implemented



<strong>[NSML](https://hack.nsml.navercorp.com/intro)</strong>

## 일정
<table class="tbl_schedule">
  <tr>
    <th style="text-align:left;width:50%">일정</th>
    <th style="text-align:center;width:15%">기간</th>
    <th style="text-align:left;width:35%">장소</th>
  </tr>
  <tr>
    <td>
      <strong>예선 1라운드</strong><br>
      2019년 1월 2일(수)~2019년 1월 16일(수)
    </td>
    <td style="text-align:center">2주</td>
    <td>
      온라인<br>
      <a href="https://hack.nsml.navercorp.com">https://hack.nsml.navercorp.com</a>
    </td>
  </tr>
  <tr>
    <td>
      <strong>예선 2라운드</strong><br>
      2019년 1월 16일(수)~2019년 1월 30일(수)
    </td>
    <td style="text-align:center">2주</td>
    <td>
      온라인<br>
      <a href="https://hack.nsml.navercorp.com">https://hack.nsml.navercorp.com</a>
    </td>
  </tr>
  <tr>
    <td>
      <strong>결선</strong><br>
      2019년 2월 21일(목)~2월 22일(금)
    </td>
    <td style="text-align:center">1박 2일</td>
    <td>
      네이버 커넥트원(춘천)<br>
    </td>
  </tr>
</table>
## 미션
* 예선 1차 : 소규모의 라인프렌즈 상품 image retrieval
* 예선 2차 / 결선(온라인, 오프라인) : 대규모의 일반 상품 image retrieval
> ※ 모든 미션은 NSML 플랫폼을 사용해 해결합니다.<br>
> &nbsp;&nbsp;&nbsp;NSML을 통해 미션을 해결하는 방법은 이 [튜토리얼](https://n-clair.github.io/vision-docs/)을 참고해 주세요.

### 예선 1차
예선 1차는 소규모의 라인프렌즈 상품 데이터를 이용한 image retrieval challenge 입니다.
Training data를 이용하여 image retrieval model을 학습하고, test시에는 각 query image(질의 이미지)에 대해 reference images(검색 대상 이미지) 중에서 질의 이미지에 나온 상품과 동일한 상품들을 찾아야 합니다.

#### Training data
Training data는 각 class(상품) 폴더 안에 그 상품을 촬영한 이미지들이 존재합니다.
- Class: 1,000
- Total images: 7,104
- Training data 예시: [training_example.zip](https://github.com/AiHackathon2018/AI-Vision/files/2719945/training_example.zip), [[참고 이슈](https://github.com/AiHackathon2018/AI-Vision/issues/33)]

#### Test data
Test data는 query image와 reference image로 나뉘어져 있습니다.
- Query images: 195
- Reference images: 1,127
- Total images: 1,322

### 예선 2차 / 결선(온라인, 오프라인)
예선 2차는 대규모의 일반 상품 image retrieval challenge 입니다.
예선 1차와 같은 방식이지만, 데이터의 종류가 라인프렌즈로 한정되어 있지 않고, 데이터의 개수가 상대적으로 큰 경우입니다.

### 평가지표
- [mAP(mean average precision)](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision)을 사용합니다.
- 동점자가 나올 경우에는 Recall@k를 계산하여 순위를 결정할 수 있습니다. [Deep Metric Learning via Lifted Structured Feature Embedding](https://arxiv.org/abs/1511.06452)

## Baseline in NSML

### Baseline model 정보
- Deep learning framework: Keras
- Docker 이미지: `nsml/ml:cuda9.0-cudnn7-tf-1.11torch0.4keras2.2`
- Python 3.6
- 평가지표: mAP
- Epoch=100으로 학습한 결과: mAP 0.0116

### NSML

1. 실행법

    - https://hack.nsml.navercorp.com/download 에서 플랫폼에 맞는 nsml을 다운받습니다.

    - `nsml run`명령어를 이용해서 `main.py`를 실행합니다.

        ```bash
        $ nsml run -d ir_ph1 -e main.py
        ```
2. 제출하기

    - 세션의 모델 정보를 확인합니다.
        ```bash
        $ nsml model ls [session]
        ```
    - 확인한 모델로 submit 명령어를 실행합니다.
        ```bash
        $ nsml submit [session] [checkpoint]
        ```

3. [web](https://hack.nsml.navercorp.com/leaderboard/ir_ph1) 에서 점수를 확인할수있습니다.

## 진행 방식 및 심사 기준
### 예선

* 예선 참가팀에게는 예선 기간중 매일 시간당 60-120 NSML 크레딧을 지급합니다.
  (누적 최대치는 2,880이며 리소스 상황에 따라 추가지급될 수 있습니다.)
* 팀 참가자일 경우 대표 팀원에게만 지급합니다.
* 사용하지 않는 크레딧은 누적됩니다.

#### ***예선 1라운드***
* 일정 : 2019. 1. 2 ~ 2019. 1. 16
* NSML 리더보드 순위로 2라운드 진출자 선정 (2라운드 진출팀 50팀 선발,순위가 낮으면 자동 컷오프)


#### ***예선 2라운드***
* 일정 : 2019. 1.16 – 2019. 1. 30
* NSML 리더보드 순위로 결선 진출자 선정 (결선 진출자 약 40팀 선발)
* 전체 인원에 따라 결선 진출팀 수에 변동이 있을 수 있습니다.

### 결선
* 일정 : 2019. 2. 21 – 2019. 2. 22 1박 2일간 춘천 커넥트원에서 진행
* 최종 우승자는 NSML 리더보드 순위(1위, 2위, 3위)로 결정합니다.
* 결선 참가자에게 제공하는 크레딧은 추후 공지 예정입니다.

> ※ 1 NSML 크레딧으로 NSML GPU를 1분 사용할 수 있습니다.<br>
> &nbsp;&nbsp;&nbsp;10 NSML 크레딧 = GPU 1개 * 10분 = GPU 2개 * 5분 사용

## github issue
[Q&A issue page](https://github.com/AiHackathon2018/AI-Vision/issues)를 통해 할 수 있습니다.<br>
