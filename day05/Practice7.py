# PythonML Practice7: 로지스틱 분류
# 데이터 출처: https://www.kaggle.com/code/anshigupta01/iris-flower-classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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

# [단계 4] 로지스틱 분류 모델 학습 (Logistic Regression)
indexs = ( train_target == 'Iris-setosa') | (train_target == 'Iris-virginica')
#print( indexs )
train_two_iris = train_scaled[ indexs ]
target_two_iris = train_target[ indexs ]
#print( train_two_iris )
#print( target_two_iris )


# [단계 5] 모델 평가 및 분류 정확도(Accuracy) 확인 * 테스트 세트의 정확도가 0.95 이상이 나오도록 설정


# [단계 6] 학습한 종속 변수 출력


# [단계 7] 테스트 세트의 앞선 5개 샘플 데이터에 대해 모델이 예측한 클래스를 출력하세요.

