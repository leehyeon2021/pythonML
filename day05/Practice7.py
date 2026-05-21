# PythonML Practice7: 로지스틱 분류
# 데이터 출처: https://www.kaggle.com/code/anshigupta01/iris-flower-classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax

# [단계 1] 데이터 로드 및 독립/종속 변수 추출
# 파일명: ./Iris.csv
df = pd.read_csv('./day01/Iris.csv')
# 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm' 4개 열을 독립 변수 X로,
x = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
# 'Species' 열을 종속 변수 y로 추출하세요.
y = df['Species']

# [단계 2] 훈련용 / 테스트용 데이터 분리
train_input , test_input , train_target , test_target = train_test_split( x , y , test_size=0.2 , random_state=42 )

# [단계 3] 데이터 표준화 (Standardization), 스케일러
ss = StandardScaler()
ss.fit( train_input )
train_scaled = ss.transform( train_input )
test_scaled = ss.transform( test_input )

# [단계 4] 로지스틱 분류 모델 학습 (LogisticRegression)
lr = LogisticRegression( C=20 , max_iter=1000 )
lr.fit( train_scaled , train_target )
print( lr.classes_ )
    # ['Iris-setosa' 'Iris-versicolor' 'Iris-virginica']
print( lr.predict( test_scaled[:3] ) )
    # 이름 ['Iris-versicolor' 'Iris-setosa' 'Iris-virginica']
print( lr.predict_proba( test_scaled[:3] ) )
    # 확률 [[2.98265012e-04 9.90178179e-01 9.52355576e-03]  -> versicolor 제일 높음
    #  [9.94578766e-01 5.42123363e-03 1.39121367e-15]       -> setosa
    #  [5.57347743e-16 1.24488666e-06 9.99998755e-01]]      -> virginica

# [단계 5] 모델 평가 및 분류 정확도(Accuracy) 확인 * 테스트 세트의 정확도가 0.95 이상이 나오도록 설정
# [단계 6] 학습한 종속 변수 출력
# [단계 7] 테스트 세트의 앞선 5개 샘플 데이터에 대해 모델이 예측한 클래스를 출력하세요.
print( lr.score( test_scaled , test_target ) )      # 1.0 (데이터가 단순정교?해서 그런듯)
decision = lr.decision_function( test_scaled[:5] )  # y=wx+b값
print( decision )
    # [[ -3.85706632   4.25059147  -0.39352515]
    #  [ 13.1383866    7.9263907  -21.0647773 ]
    #  [-18.88340577   2.64347048  16.2399353 ]
    #  [ -4.18679516   3.63657903   0.55021614]
    #  [ -6.66751669   4.9147544    1.75276229]]
print( softmax( decision ) )
    # [[1.78963159e-09 5.94120690e-06 5.71426602e-08]
    #  [4.30323495e-02 2.34560025e-04 6.01935159e-17]
    #  [5.33221104e-16 1.19099763e-06 9.56710504e-01]
    #  [1.28695754e-09 3.21523307e-06 1.46832465e-07]
    #  [1.07696242e-10 1.15429542e-05 4.88743785e-07]]
print( np.round( softmax( decision ) ) )
    # [[0. 0. 0.]
    #  [0. 0. 0.]
    #  [0. 0. 1.]
    #  [0. 0. 0.]
    #  [0. 0. 0.]]
print( np.round( softmax(decision) , decimals=4) )
    # [[0.    0.    0.   ]
    #  [0.043 0.    0.   ]
    #  [0.    0.    0.957]
    #  [0.    0.    0.   ]
    #  [0.    0.    0.   ]]