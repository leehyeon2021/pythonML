
# 결정 트리 , 트리 노드

import pandas as pd
df = pd.read_csv('./day06/wine.csv')
#print( df.head(10) ) 

# alcohol , sugar , pH , class

# 1. 와인 정보 불러오기
data = df[ ['alcohol','sugar','pH']]    # 와인의 속성 3개
target = df['class']                    # 1:화이트와인 0:레드와인

from sklearn.model_selection import train_test_split
train_input , test_input , train_target , test_target = train_test_split( data, target, random_state=42 )

# 2. 결정 트리 (분류 모델)
from sklearn.tree import DecisionTreeClassifier     # 의사결정 트리 분류
dt = DecisionTreeClassifier()                       # 모델 객체 생성
dt.fit( train_input , train_target )                # 모델 학습
print( dt.score( train_input , train_target ) )     # 0.9973316912972086
print( dt.score( test_input , test_target ) )       # 0.8504615384615385
# 모델 예측
print( dt.predict( test_input[:5] ) )               # [1. 0. 1. 1. 1.]

# 3. 결정 트리 시각화
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree                  # 트리 시각화
plot_tree( dt , max_depth= 1 , feature_names=['alcohol','sugar','pH'], class_names=['Red Wine', 'White Wine'], filled=True)
    # plot_tree( 트리모델 , max_depth=가지수 , feature_names= )
        # max_depth=트리의 
plt.show()

# 트리: 전체적인 구조 그 자체를 의미
# 노드: 사각형 상자 하나하나를 의미. 가장 위에 있는 노드는 루트(root) 노드
# 노드 속성: 
    # - `value=[예측타겟수]` : [ 85 , 2097 ] 0으로 예측한 수는 85개 / 1로 예측한 수는 2097개라는 뜻
    # - `gini=불순도` : 0.075  
        # - 0에 가까울수록 순수(특정 예측값으로 모여)하다.
        # - 0.5에 가까울수록 혼란하다.
    # - `suger=특성` : suger<=4.15보다 작으면 true(왼쪽노드로이동), false(오른쪽노드로이동)

# 4. 특성 중요도
# `feature_importances_`: 각 특성이 트리 모델에 얼마나 중요한 역할을 하는지에 대한 수치 (합은 1)
print( dt.feature_importances_ )        # [0.23054233 0.51947326 0.24998442]
print( dt.feature_importances_[0] )     # 0.23786262514675469
                                        # alcohol sugar pH

# 5. 최소한의 불순도(gini) 설정: 최적의 파라미터
dt = DecisionTreeClassifier( random_state=42 , min_impurity_decrease=0.0005 )
dt.fit( train_input , train_target )
print( dt.score( train_input , train_target ) )     # 0.8975779967159278 : 
print( dt.score( test_input , test_target ) )       # 0.8590769230769231 : 과대 적합 최소화