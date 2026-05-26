

#  교차검증(cross_validate), 그리드 서치(GridSearchCV), 랜덤 서치(RandomizedSearchCV)

# * 그래픽카드 사양 중요. 학습 시엔 코랩 사용

import pandas as pd
df = pd.read_csv('./day06/wine.csv')
#print( df.head(10) ) 

# alcohol , sugar , pH , class

# 1. 와인 정보 불러오기
data = df[ ['alcohol','sugar','pH']]    # 와인의 속성 3개
target = df['class']                    # 1:화이트와인 0:레드와인

from sklearn.model_selection import train_test_split
train_input , test_input , train_target , test_target = train_test_split( data, target, random_state=42 )

# 2. 결정 트리 (분류모델) (전처리 중요함)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier( random_state=42 )
dt.fit( train_input , train_target )
print( dt.score( test_input , test_target ) )       # 결정트리만 했을 때: 0.8516923076923076

# 3. 교차 검증
from sklearn.model_selection import cross_validate
# cross_vlidate( 학습모델 , 학습세트 , 정답세트 )
# 교차검증은 전체 데이터를 N등분(폴드)하여 돌아가면서 검증 한다. 기본값은 5등분
# 즉) 데이터를 여러 조각으로 나누어 학습하는 방법
scores = cross_validate( dt , train_input , train_target )
print( scores )
# {'fit_time': array([0.00566888, 0.00536299, 0.0054729 , 0.00509048, 0.00541735]),
#  'score_time': array([0.00127625, 0.00157619, 0.00155854, 0.0012877 , 0.00146437]),       혼자 다 하네
#  'test_score': array([0.85128205, 0.84820513, 0.8788501 , 0.85112936, 0.84394251])}       대박
import numpy as np
print( np.mean( scores[ 'test_score' ] ) )  
                                                    # 5등분 학습의 평균 점수: 0.8546818301479492

from sklearn.model_selection import StratifiedKFold
# StratifiedKFold( n_splits=N등분 )     (마찬가지로 기본값은 5등분)
splits = StratifiedKFold( n_splits=10 , shuffle=True , random_state=42 )
scores = cross_validate( dt , train_input , train_target , cv=splits )
print( scores )     # 10등분 했으니 10개
# , 'test_score': array([0.86680328, 0.84836066, 0.88090349, 0.83983573, 0.8788501 ,
#       0.85420945, 0.84804928, 0.85626283, 0.84804928, 0.86447639])}
print( np.mean( scores['test_score'] ) )
                                                    # 10등분 학습의 평균 점수: 0.8585800484734237   (아주 조금 증가하긴 함)

# 4. 그리드 서치: 최적의 파라미터(변수/학습에 필요한 설정값)
from sklearn.model_selection import GridSearchCV
# 1) 여러 개 '최소불순도' 설정 : 임의의 최소 불순도 넣어서 리스트로 구성
# 불순도란?: 0에 가까울수록 예측값이 명확하다. (과대적합)
#           0.5에 가까울수록 예측값이 애매하다.
params = { 'min_impurity_decrease' : [ 0.0001 , 0.0002 , 0.0003 , 0.0004 , 0.0005 ] }
# 2) 연산 : `GridSearchCV( 트리모델 , {파라미터들} , n_jobs=-1 )`
# `n_jobs=-1`: 컴퓨터 내 모든 CPU 코어 사용하여 병렬(스레드) 연산
    # -> CPU 최대 사용 잡아먹음
gs = GridSearchCV( DecisionTreeClassifier( random_state=42 ) , params , n_jobs=-1 )
# 3) 그리드 서치 학습
gs.fit( train_input , train_target )    # 기본값으로 교차검증 5(번) 적용.
dt = gs.best_estimator_                 # 최적의 파라미터로 학습 결과 보기
print( dt.score( test_input , test_target ) )       # 0.8670769230769231
print( gs.best_score_ )                             # 0.8731517927657558
print( gs.best_params_ )                        # {'min_impurity_decrease': 0.0003}
print( gs.cv_results_ ) # 결과 많이 나오는데 split 5번 된다는 거 확인.
                        # 기본값으로 교차검증 5(번) 적용.

# 5. 다중 파라미터
params = {
    # 최저불순도: 0.0001 ~ 0.001 (미만까지) 0.0001씩 증가 -> 9번
    'min_impurity_decrease': np.arange( 0.0001 , 0.001 , 0.0001 ),  # 9번
    # 최대 뿌리 길이: 5~20(미만까지) 1씩 증가
    'max_depth': range( 5, 20, 1 ),                                 # 15번
    # 노드 분할 시 최저 샘플 수: 최저 샘플 수보다 적으면 노드 분할 안 함.
    'min_samples_split': range( 2, 100, 10 ),                       # 10번
    # 리프노드(나머지 뿌리/노드) 최저 샘플 수: 현재 리프노드가 최저 샘플 수보다 작으면 노드 분할 안 함
    'min_samples_leaf': range( 1, 100, 10 )                        # 10번
}
# cv = 교차검증수(N등분). 기본값은 5
# n_job=-1: CPU 병렬 처리 수행하라는 뜻
gs = GridSearchCV( DecisionTreeClassifier( random_state=42 ) , params , n_jobs=-1 , cv=5 )
# 대략 학습 조합
# 최저불순도(9가지) * 깊이(15가지) * 최저분리샘플(10가지) * 최저리프샘플(10가지) = 대략 13,000가지 조합을 학습한다.
# +) + 교차검증(cv, N등분) * 대략 13,000가지 조합 = 6만 번의 학습 모델
gs.fit( train_input , train_target )
# 최적의 파라미터 조합
print( gs.best_params_ )
# {'max_depth': 13, 'min_impurity_decrease': np.float64(0.0001),
#  'min_samples_leaf': 11, 'min_samples_split': 2}
print( gs.best_score_ )                             # 0.8756162796819881 : 조금 증가

# 6. 랜덤 서치
# 랜덤 서치란?: 고정된 값이 아니라 '확률 분포 함수'를 제공하여 무작위로 숫자를 뽑아 학습한다.
# 조합 수가 많아지면 연산량이 많아져서 서버(컴퓨터)에 부하 발생할 수 있다.
from sklearn.model_selection import RandomizedSearchCV
rs = RandomizedSearchCV( DecisionTreeClassifier( random_state=42 ), params , n_iter=100, n_jobs=1 , cv=5 , random_state=42 )
    # `n_iter=100` : 정의된 조합 수에서 무작위(랜덤)으로 N개의 조합만 추출하여 학습한다.
# 대략 13,000개 조합에서 100개만 무작위로 추출: 교차검증5 -> 500번 학습
rs.fit( train_input , train_target )
print( rs.best_params_ )
# {'min_samples_split': 12, 'min_samples_leaf': 11,
# 'min_impurity_decrease': np.float64(0.0004), 'max_depth': 17}
print( rs.best_score_ )                             # 0.8694571684304744 : 학습 속도는 빨라졌으나 정확도는 조금 낮아짐.

# 최고의 파라미터로 예측
# dt = rs.best_estimator_
# print( dt.predict( 임의의예측값아무거나 ) ) 이후에 이렇게 예측해나가면됨