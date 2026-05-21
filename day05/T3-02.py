

# 로그 로스
# 확률과 오차값을 보여줌

# 1. 여러가지 특성에 따른 분류 모델
import pandas as pd
df = pd.read_csv('./day01/Fish.csv')
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

# +) 확률적인 검사 하강법
# fit() 모델학습에서는 정답(target)도 같이 학습 중이다. 예측(y)값과 실제 정답 간의 오차(기울기/가중치(w) 조절 -> 오차 줄이기) 측정
# 예) 산꼭대기에서 내려가는 방법 중에 가장 최적의 경로로 내려오는 방법 = 겅사 하강법( 수많은 경우의 수 계산하여 판단 )
# 1) 전통 경사 하강법: 정확도가 좋지만 학습 속도가 느리다.
# 2) 확률 경사 하강법(SGD): 정확도는 낮지만 학습 속도가 빠르다. 미니배치

# +) 로그 로스 (손실 함수)
# 손실: 예측과 정답의 전체적인 차이
# 로그 로스 함수는 0과 1의 확률 값이 아닌 **오차 값**을 측정
# 경사 하강법과 같이 사용될 수밖에 없다.

# +) 에포크
# 학습 횟수

# 2. SGDClassifier : 분류 모델
from sklearn.linear_model import SGDClassifier
# 모델 객체 생성
# loss = 'log_loss' : 로스 함수
# `random_state`: SGD가 전체 데이터 학습이 아닌 일부 자료(매니배치)를 가지고 학습하는 데 사용되는 분리 기준(난수값)
# `max_iter`: (반복) 계산 횟수. 미니배치이므로 전체 데이터셋을 '10'이면 10 반복 학습하여 모델 성공 향상 / 최적의 정확도에서 멈춤 ( 에포크 )
# `tol=None`: 쵲거의 정확도를 찾아도 계속 반복 학습하도록 설정
sc = SGDClassifier( loss= 'log_loss', random_state=42, max_iter=10, tol=None)
# 모델 학습
sc.fit( train_scaled , train_target )
print( sc.score( test_scaled , test_target ) )
print( sc.predict( test_scaled[ : 3 ]) )            # ['Perch' 'Smelt' 'Pike']

# 3. 점진적(부분) 학습: 중간에 일부를 다시 학습 가능하다.
sc.partial_fit( train_scaled , train_target )       # (위에서이미학습된모델어)10번 + 1번 => 11번 학습
print( sc.score( test_scaled , test_target ) )      # 0.75 과적합 때문에 정확도 떨어질 수 있음.

# 4. 최적의 에포크(학습(반복)횟수) 찾기
sc = SGDClassifier( loss='log_loss', random_state=42 )  # `max_iter` 생략 시 1번 학습

train_score = []    # 학습용 정확도
test_score = []     # 테스트용 정확도

import numpy as np
classes = np.unique( train_target )     # 정답지에서 중복을 제거하고 고유 정답만 추출

for i in range( 0 , 100):   # 100번 반복 (에포크(x축) 바뀜)
    sc.partial_fit( train_scaled, train_target, classes=classes )                # 1번 학습
    train_score.append(sc.score( train_scaled, train_target ))
    test_score.append(sc.score( test_scaled, test_target ))

# 정확도 시각화
import matplotlib.pyplot as plt
plt.plot( train_score )     # 학습용 정확도 점수            
plt.plot( test_score )      # 테스트용 정확도 점수
# 둘이 너무 붙어있으면: 과소적합
# 둘이 멀어지면: 과대적합
# 학습용과 테스트용이 고르게 오르는 시점: 최적. 적합
# (에포크(x축)를 조절하면서 보기)
plt.show()