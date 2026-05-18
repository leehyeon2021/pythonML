
# 1차원 -> 2차원 변환
# T1-01( zip활용 ), T1-02( column_stack활용 ), T2-01( reshape활용 )


# 1. 가져오기
import pandas as pd
df = pd.read_csv('./day01/Fish.csv')
df.info()

# 2. Perch(농어)만 추출
target_fish = df[ df['Species'].isin(['Perch'])]
target_fish.info()  # 56마리
# 농어의 길이/무게 추출
perch_length = target_fish['Length2'].values
perch_weight = target_fish['Weight'].values

# 농어 '길이'에 따른 '무게' 예측
import matplotlib.pyplot as plt
plt.scatter( perch_length , perch_weight )
plt.show()

# 3. 학습 모델 만들기
# 1) 준비: 학습용(train)과 테스트용(test) 분리. -왜?-> 모델평가에 사용하려고.
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split( perch_length, perch_weight, test_size=0.3, random_state=42 )
    # `random_state=분리할 때 사용되는 난수값`: 난수값에 따라 분리한다. 0~32억 사이 아무 값이나 넣어주면 된다.
    # 고정값을 넣어주면 항상 동일한 분리 값을 넣을 수 있다.
# 2) 자료형식(모양) 구성: (사이킷런)모델 학습 전에 형식 만들기. 대부분 2차원 구성
import numpy as np
# `shape`: 배열의 모양을 반환해준다.
array = np.array( [1, 2, 3, 4] )    # 1차원: (4,) <-(행 , 열)
print( array.shape )
array2 = np.array([[1, 2],[3, 4],[5, 6]])   # 2차원: (3, 2)
print( array2.shape )
print( train_input.shape )  # (39,) 1차원 배열 나옴 -> 사이킷런 모델들은 1차원배열 학습이 불가능하다.
print( train_input )        # 1차원으로 구성된 '농어' 길이

# 4. 1차원 -> 2차원
# T1-01( zip활용 ), T1-02( column_stack활용 ), T2-01( reshape ) 1차원->2차원
# `.reshape( 행개수, 열개수 )`: 행개수에는 -1 넣음 (-1은 자동으로 해달라는 뜻(자료개수만자동)), 열 개수는 1개
train_input = train_input.reshape( -1, 1 )  # 2차원으로 바뀜!
print( train_input )  # [17.4 36.  25.  40. (생략)] -> [[17.4] [36. ] [25. ] [40. ] (생략) ]
test_input = test_input.reshape( -1, 1 )

# 5. 모델 학습
from sklearn.neighbors import KNeighborsClassifier  # K최근접이웃 모델: 주변 이웃의 종류를 봄
from sklearn.neighbors import KNeighborsRegressor   # K최근접이웃 회귀: 주변 이웃의 숫자를 봄
knr = KNeighborsRegressor()                         # 모델 객체 생성
knr.fit( train_input , train_target )               # 모델 학습 (길이, 무게) <- '길이'에 따른 '무게' 학습
print( knr.score(test_input, test_target))          # 모델 평가
print( test_input )
print( knr.predict( test_input ) )

# 6. k최근접이웃 회귀는 이웃의 평균으로 예측 한다. 하이퍼라미터(k) 조절
# k = 이웃 개수 정하기
knr = KNeighborsRegressor()                         # 모델 객체 생성
# 임의의 길이 생성: 임의의 물고기 길이 5부터 45까지 생성 (45개 임의값)
x = np.arange( 5 , 45 ).reshape( -1 , 1 )
print( x )  # 5 ~ 44 까지의 임의의 값
for k in [ 1 , 3 , 5 , 10 ]:                        # 이웃 개수를 4가지 모델로 학습
    knr.n_neighbors = k                             # 현재 모델의 이웃개수 대입
    knr.fit( train_input , train_target )           # 총 4번 학습예정
    print( knr.score(test_input , test_target) )    # 총 4번 학습평가
    pred = knr.predict( x )                         # 임의의 값 x(물고기 길이)로 예측
    print( pred )                                   # 총 45개의 물고기 길이의 몸무게를 예측한다.
    # 시각화
    plt.scatter( train_input , train_target )
    plt.plot( x , pred )                            # plot ( 선차트이면서 회귀(예측)선 ) x=길이 pred=몸무게(예측)
    plt.show()

# k는 이웃개수 뜻한다. k최근접 회귀는 이웃의 평균으로 예측한다.
# k가 1일 때   0.9918926744767643       -> 과대 적합 훈련: 특정한 자료에 튀는 데이터(노이즈/이상치)까지 적용될 수 있으므로 예측이 망가질 수 있다.
# k가 3일 때   0.9766857219041255
# k가 5일 때   0.9929281790592219   -> 균형적인 추세 표현
# k가 10일 때  0.9742254836937329       -> 과소 적합 훈련: 많은 자료에 둔감한 단순화된 자료까지 적용될 수 있으므로 예측이 망가질 수 있다.
# k가 5일 때 가장 균형적인 추세 표현이다. 회귀선이 너무 꺾이거나 완만한 일직선이 아니다.
    # 그래프가 확 꺾여 뾰족한 것 보다는 완만한 것이 좋대
        # 1일 때는 너무 직선이 너무 많아 뾰족해서 x , 10일 때는 직선이 너무 많아 일직선이라서 x

# 결론: 머신러닝에서는 가장 최적의 파라미터(k/이웃개수)를 찾는 과정을 한다. 이를 '튜닝'이라 한다.