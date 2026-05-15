
# 캐글의 데이터셋: https://www.kaggle.com/datasets/vipullrathod/fish-market

# 1. csv 불러오기
import pandas as pd
df = pd.read_csv('./day01/Fish.csv')
print( df.head() )


# 2. 특정한 물고기 추출: 도미(Bream)의 길이, 무게 추출
bream_df = df[ df['Species'] == 'Bream']

bream_length = bream_df['Length2'].tolist()
bream_weight = bream_df['Weight'].tolist()
print( bream_length, bream_weight )


# 3. 특정한 물고기 추출: 빙어(Smelt)의 길이, 무게 추출
smelt_df = df[ df['Species'] == 'Smelt' ]

smelt_length = smelt_df['Length2'].tolist()
smelt_weight = smelt_df['Weight'].tolist()
print( smelt_length, smelt_weight )


# 4. 시각화
import matplotlib.pyplot as plt
plt.scatter( bream_length, bream_weight )       # 도미 시각화
plt.scatter( smelt_length, smelt_weight )       # 빙어 시각화
plt.xlabel('length(CM)')
plt.ylabel('weight(gram)')
plt.show()


# 5. 도미와 빙어 자료 합치기: 길이끼리, 무게끼리
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

# 6. 2차원리스트 -> [[길이, 무게],[길이, 무게]]
# zip( 1차원리스트, 1차원리스트 ): 두 리스트를 요소 하나씩 반복
# 리스트내포: [표현식 for 반복변수 in 반복값 if 조건식]
fish_data = [ [l, w] for l , w in zip(length, weight) ]
print( fish_data )
#print( bream_df.info )  # 도미 35마리
#print( smelt_df.info )  # 빙어 14마리

# 7. target(==정답지) 만들기: 1-도미,35개만듦 , 2-빙어,14개만듦 (5번에서 도미+빙어 순서로 해놔서.)
fish_target = [1]*35 + [0] * 14
print( fish_target )    # 왜 함? => 컴퓨터가 예측하도록 만들기 
# [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] => 도미:1 , 빙어:0

# 8. 알고리즘 모델
# 1) K-최근접 이웃(K-NN알고리즘): 임의값을 넣었을 때 기존 값들 중에 가장 가까운 값 찾기
# - 설치: 사이킷런( 다양한 머신러닝 모델 제공 ) `py -m pip install scikit-learn` (http://scikit-learn.org/stable/)
# 2) K-NN 모델 호출
from sklearn.neighbors import KNeighborsClassifier
# 3) K-NN 모델 객체 생성
kn = KNeighborsClassifier()
# 4) K-NN 학습하기: 문제와 답을 같이 줌 --> <지도학습: 문제와 정답을 알려주면>
# - 컴퓨터에게 미리 문제(자료) 제공하고 그 문제에 따른 답(자료)를 제공함으로써 기억한다.
kn.fit( fish_data, fish_target )    # fish_data: 도미와 빙어 자료 , fish_target: 도미인지 빙어인지 식별 자요
# 5) 학습된 모델의 점수(정확도) 측정: `kn.score(문제, 답)` 0~1사이값으로 반환, 1이면 100점
print( kn.score( fish_data , fish_target ) )    # 1.0
#print( int(kn.score( fish_data , fish_target ) *100) , '점') #그냥 해봄
# 6) 임의의 값 넣어서 예측 측정: `kn.predict( [임의값] )`
print( kn.predict( [ [30, 600] ] )) # 임의의 물고기 길이cm와 무게600g -> '도미? 빙어?' 예측.
# 7) 임의의 값 시각화
plt.scatter( bream_length, bream_weight )       # 도미 시각화
plt.scatter( smelt_length, smelt_weight )       # 빙어 시각화
plt.scatter( 30, 600 )  # 임의의 값 위치 한 번 확인해보기. (나중에 배우는 머신러닝과 관련)
plt.xlabel('length(CM)')
plt.ylabel('weight(gram)')
plt.show()
# 8) 근접한 이웃 찾을 기준 정하기: 하이퍼파라미터
# KNeighborsClassifier( n_neighbors= 참고할이웃개수 )
# 현재 예제는 도미35, 빙어14 -> 총49마리 (전체)
kn = KNeighborsClassifier( n_neighbors= 49 )    # 접근할 49개 중에서 정답 찾기
kn.fit( fish_data , fish_target )               # 학습
print( kn.score( fish_data , fish_target ) )    # 정확도 측정 -> 0.7142857142857143(71점) -> 빙어는 오답처리
# 총 49마리 중에서 참조할 이웃을 49마리로 설정하면 어떠한 임의의 값 예측 하더라도 무조건 도미 수가 많아서 '도미'
# 추후에 방대한 데이터로 활용 시 최적의 k값 찾기 --> 딥러닝