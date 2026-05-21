
# 로지스틱 회귀 모델 ( 이진분류 , 다중분류 )
# 시그모이드 함수가 중요 역할, 확률을 보여줌.

# 1. 여러가지 특성에 따른 분류 모델
import pandas as pd
df = pd.read_csv('./day01/Fish.csv')
print( df.head(20) )
# 어종 7개: Species
fish_target = df[ 'Species' ]
# 특성 6개: Weight,Length1,Length2,Length3,Height,Width
fish_input = df[ ['Weight','Length1','Length2','Length3','Height','Width']]
# 훈련/테스트 분리
from sklearn.model_selection import train_test_split
train_input , test_input , train_target , test_target = train_test_split( fish_input , fish_target , test_size=0.25 , random_state=42 )
# 스케일링
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit( train_input )
train_scaled = ss.transform( train_input )
test_scaled = ss.transform( test_input )

# 로지스틱 회귀 = 이진분류 = 시그모이드 함수(공식)
# 선형 방정식의 출력값을 0과 1(확률/분류) 사이의 값으로 변환해주는 공식/함수
# 사용처) 암 발병 확률 / 스팸 메일 분류 등: 이진분류 알고리즘을 사용한다.
# 즉) 컴퓨터는 수치상의 150 또는 -82.3 (수치)값으로 말하는 확률이라 어려움
    # 확률이란?: 항상 0(0%)에서 1(100%) 사이어야 한다.
import numpy as np
import matplotlib.pyplot as plt
z = np.arange( -5 , 5, 0.1 )    # -5부터 5까지 0.1씩 증가하는 리스트
    # 시작값(-5)과 마지막 값(5)이 바뀌어도 y값은 같다.(당연하지만확률이라) S모양인 것도 같다. (Sㅣ그모이드)
phi = 1 / ( 1 + np.exp( -z ) )  # 시그모이드 공식
plt.plot( z, phi )              # 시그모이드 시각화
plt.show()


# 2. 이진 분류: 로지스틱 회귀 모델(`LogisticRegression`)
# 이진 분류: 0 또는 1로 분류하는 방법
indexs = ( train_target == 'Bream') | (train_target == 'Smelt')     # 도미와 빙어만 추출
print( indexs )
train_bream_smelt = train_scaled[ indexs ]
target_bream_smelt = train_target[ indexs ]
print( train_bream_smelt )
print( target_bream_smelt )
# 이진분류 모델 구현
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit( train_bream_smelt , target_bream_smelt )                    # 도미와 빙어만 학습
# 이진분류 모델 예측
print( lr.predict( train_bream_smelt[ : 3] ) )                      # 3개만 예측       ['Bream' 'Smelt' 'Bream']
print( lr.predict_proba( train_bream_smelt[ : 3] ) )                # 3개만 예측 확률   [[도미확률, 빙어확률]] -> 총합확률은1(100%)
# 임계값은 0.5 기준으로: 0.5 이상이면 도미로 예측 / 0.5 미만이면 빙어 예측한다.
# [[0.95684973 0.04315027]  -> 도미일 확률
#  [0.99825845 0.00174155]  -> 
#  [0.02395382 0.97604618]] -> 

# 3. 다중 분류: 로지스틱 회귀 모델
# 하이퍼파라미터
# C: 규제를 완화하여 릿지/라쏘 모델처럼 정확도 설정 가능하다. 
    # 모델의 성능을 향상하기 위해 가중치 값들을 자동으로 조정
    # max_iter: 다중분류 계산 횟수
        # (생략시) 기본값 100으로 최적의 정확도를 찾을 때까지 계산 반복횟수 조정 (넉넉하게)
lr = LogisticRegression( C=20 , max_iter=1000 )
lr.fit( train_scaled , train_target )   # 모든 어종 학습
# 모델 예측
print( lr.predict( test_scaled[: 3]) )          # 3개만 예측 ['Perch' 'Smelt' 'Pike']
print( lr.predict_proba( test_scaled[ : 3]) )   # 3개만 예측 확률: 분류개수 만큼의 확률
# 모델 평가: 선형 회귀와 다르게 **결정계수**라고 하지 않고 맞힌 **비율(정확도)**를 반환한다고 한다.
print( lr.score( test_scaled , test_target ) )  # 0.85
# 소프트맥스
from scipy.special import softmax
decision = lr.decision_function( test_scaled[ : 3] )
print( decision )                               # 0.925
print( softmax( decision ) )
    # # `softmax` 함수로 결과값을 확인했을 때 predict와 동일하게 출력된다.
print( np.round( softmax( decision ) , decimals= 3 ) )  # np.round( 값 , decimals= 소수점)
# [[0.    0.001 0.041 0.    0.007 0.    0.   ]
#  [0.    0.001 0.031 0.    0.005 0.688 0.   ]
#  [0.    0.    0.007 0.212 0.002 0.006 0.   ]]
# 다중 분류의 확률을 검증할 때는 `.classes_` 종속변수들의 순서 확인
print( lr.classes_ )    # ['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']