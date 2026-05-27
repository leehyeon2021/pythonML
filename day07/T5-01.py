
# 회귀분석/분류분석/k-최근접 -> 지도학습(정답있음)
# 군집분석 -> 비지도학습(정답없음)

# ==========================================
import pandas as pd
# 샘플 데이터 준비 (3차원 특성: 무게, 당도, 단단함)
data = {  
    'weight': [110, 160, 130, 320, 370, 300, 55, 65, 60, 210, 220, 200, 90, 80, 100, 190, 180, 170, 100, 90,
               140, 280, 320, 130, 200, 140, 250, 150, 70, 80, 200, 300, 220, 140, 180, 230, 220, 250],
    'sweetness': [6.2, 7.2, 6.8, 8.1, 8.6, 8.1, 5.2, 5.7, 6.1, 7.2, 7.6, 6.7, 7.3, 6.9, 7.3, 7.5, 7.4, 7.3, 7.0, 6.8,
                  6.9, 8.0, 8.1, 6.7, 7.0, 6.6, 7.8, 7.1, 6.7, 6.5, 7.0, 7.6, 7.3, 7.0, 7.2, 7.5, 7.4, 7.7],
    'hardness': [7.8, 6.5, 7.1, 4.2, 3.5, 3.9, 8.9, 8.4, 8.1, 5.8, 5.2, 6.1, 7.3, 7.5, 7.0, 5.9, 6.2, 6.4, 7.2, 7.6,
                 6.8, 4.5, 4.1, 7.0, 5.7, 6.9, 4.9, 6.6, 8.2, 8.5, 5.8, 4.0, 5.3, 6.7, 6.1, 5.0, 5.2, 4.7]
}
df = pd.DataFrame(data)

# 테스트용
newDf = pd.DataFrame({'weight': [110], 'sweetness': [7.0], 'hardness': [7.5]})
features = ['weight', 'sweetness', 'hardness']
# ==========================================

# 1. 
from sklearn.cluster import KMeans
# `n_clusters=k`: 그룹 수 설정 2이면 2개의 그룹으로 군집화한다.
# `random_state`: 그룹/군집/클러스터 설정하기 위한 초기 중심점 무작위 난수 생성 값(시드)
km = KMeans( n_clusters=2 , random_state=42 )          # 모델 객체 생성
km.fit( df[ features ] )                # 모델 학습 ( target 없음! )
        # 모델 비지도 학습 : target(정답/레이블)이 없다.
print( km.predict( newDf[features] ) )  # [0] : 모델 예측(클러스터/군집화)
print( km.labels_ )                     # 행마다의 군집 번호
    # n_clusters 없음: [0 3 5 2 6 2 4 4 4 1 1 1 0 4 0 3 3 3 0 0 5 7 2 5 1 5 7 5 4 4 1 2 1 5 3 1 1 7]
    # n_clusters= 2 : 0과1 2개 [0 0 0 1 1 1 0 0 0 1 1 1 0 0 0 1 1 0 0 0 0 1 1 0 1 0 1 0 0 0 1 1 1 0 1 1 1 1]

# 시각화
import matplotlib.pyplot as plt
plt.scatter( df['weight'] , df['sweetness'] , c = km.labels_ )
plt.scatter( newDf['weight'] , newDf['sweetness'] , marker='^' )
plt.show()

# 특성들 간 서로 다른 단위의 의미 => 스케일링
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
scaledDf = ss.fit_transform( df[features] )      # fit + trasform
scaledNewDf = ss.transform( newDf[features] )
# 스케일링 이후 시각화
plt.scatter( scaledDf[:,0] , scaledDf[:,1] , c=km.labels_ )
plt.scatter( scaledNewDf[:,0] , scaledNewDf[:,1] , marker='^' )
plt.show()


# 2. 최적의 k(그룹 수) 찾기: 엘보우 방법(오차 측정)
sse = [ ]       # 오차 저장 리스트
for k in range( 1 , 11 ):
    km = KMeans( n_clusters=k , random_state=42 )   # k 개수만큼 클러스터가 존재하는 모델 생성
    km.fit( scaledDf )                              # 스케일링 된 자료로 학습
    sse.append( km.inertia_ )                       # `inertia_`: 군집/그룹/클러스터 내 자료들 간 오차의 제곱합 측정
print( sse )
# [114.0, 44.53541576530211, 24.558046435902174, 14.458485801294604, 12.419855401356509, 9.762237329809395, 7.399463438045369, 6.391996332917253, 4.949916305626276, 4.432004354875466]

# 오차 시각화
plt.plot( range( 1 , 11 ) , sse , marker='o' )
plt.show()

# 엘보우 포인트: SSE(오차의제곱합)의 차이가 급격하게 줄어든 포인트 => 최적의 K
    # => 팔꿈치처럼 꺾이는 부분을 최적의 K로 선정하면 된다. 완만한 곳x
km = KMeans( n_clusters=3 , random_state=42 )
km.fit(scaledDf)
df['cluster'] = km.labels_      # 클러스터 결과물
# weight , sweetness , hardness , cluster


# 3. 거리 예측/계산(추론계산식): 유클리드 거리
import numpy as np
# 1) 클러스터(3개)들의 중심점
centerClus = km.cluster_centers_
print( centerClus )
# 2) 중심점에서 새로운자료의 오차(차이) 계산: 오차합의 제곱. 제곱근 씌우기.
    # np.sum( 리스트 , axis=축기준 ) . 축기준=0은 열, 1은 행
print( np.sum([1,2,3]))                         # 6
print( np.sum([[1,2,3],[4,5,6]] , axis=1))      # [ 6 15]
print( np.sum([[1,2,3],[4,5,6]] , axis=0))      # [5 7 9]
# `np.sqrt( 리스트 )`: 제곱근(루트)
result = np.sqrt(np.sum(( centerClus - scaledNewDf )**2 , axis=1))
# 3) 거리 확인
print( result )         # 클러스터 중심점에서 새로운 자료까지의 거리
# [1.26432524 3.25608991 0.97575994]
print( km.predict(scaledNewDf) )    # 2에서 한 유클리드 거리 계산과 predict 예측과 동일하다.


# 4. GMM: 가우시안 모델. 군집확률
from sklearn.mixture import GaussianMixture
gm = GaussianMixture( n_components=3 , random_state=42 )    # 객체 생성
    # KMeans와 유사하게 `n_components`로 컴포넌츠(위에서는클러스터) 설정
gm.fit( scaledDf )                                          # 학습
print( gm.predict( scaledNewDf ) )
print( gm.predict_proba( scaledNewDf ) ** 100 )
# 시각화
plt.scatter( scaledDf[:,0] , scaledDf[:,1] , c=df['cluster'] )
plt.scatter( scaledNewDf[:,0] , scaledNewDf[:,1] , marker='^' )
plt.show()
# 현재 특성이 3개 이므로 3D차원 시각화 필요 -> N차원(특성많으면) 시각화는 힘들다.


# 5. PCA: 차원 축소 , 주성분 분석 목적. 차원이 크면 시각화가 불가능하다. 주로 2차원/3차원 압축한다.
    # => PCA는 여러 개 성분을 하나로 해주는 역할
from sklearn.decomposition import PCA
# 여러 개 특성/성분을 가진 모델들을 2/3 차원으로 변경
pca = PCA( n_components=2 )     # 객체 생성: 주로 2 또는 3으로 사용된다.
# 주성분 만들기: 각 특성/성분 마다의 가중치 터해서 데이터 변동성 계싼
# 예) pca = 무게*가중치1 + 당도*가중치2 + 단단함*가중치3
pcaDf = pca.fit_transform( scaledDf )
print( pcaDf )      # 행 = 데이터 수 / 열 = 주성분 수
df['pca_x'] = pcaDf[:,0]    # 첫 번째 열을 제1주성분  ->  데이터의 변동성을 가장 크게 설명하는 주성분
df['pca_y'] = pcaDf[:,1]    # 두 번째 열을 제2주성분  -> 제1주성분을 직교하면서 두번째로 변동성(가중치)을 가장 크게 설명하는 주성분
# 주성분의 가중치 확인
compoenets = pca.components_
print( compoenets )         # 무게가중치, 당도가중치, 단단가중치
# 예측할 값도 주성분으로 변경 ((X) pca_1은 무게고 pca_2는 단단함이다 라고 해석하면 안 됨)
    # 변경하는 이유: 앞전엔 임의로 시각화 한 거지, 군집의 시각화를 위해선 특성을 표현하기 위해 pca를 사용해 변경해줘야 한다.
pcaNewDf = pca.transform( scaledNewDf )
# 시각화
plt.scatter( df['pca_x'] , df['pca_y'] , df['cluster'] , cmap='virids' , marker='o')
plt.scatter( pcaNewDf[: , 0] , pcaNewDf[: , 1] , marker='^')            # 예측할 값
plt.xlabel('pca_1')
plt.ylabel('pca_2') 
plt.show()


# 6. 분석 모델 스코어: 실루엣 스코어( 분리/응집 평가 )
from sklearn.metrics import silhouette_score
# `silhouette_score( 자료 , 모델군집 )`
sc = silhouette_score( scaledDf , km.labels_ )              # k-mean 평가 모델
print( sc )                                                 # 0.443044585687068

# gm 가우시안 모델에서는 k-mean처럼 lables_ 속성이 없다.
    # => 그래서 예측을 통해 군집도를 구한다.
sc = silhouette_score( scaledDf , gm.predict( scaledDf ) )  # GMM 평가 모델 
print( sc )                                                 # 0.46537942714394775



# 스코어 개선
# 1. 최적의 K
# 2. PCA 주성분(가중치)
# 3. 이상치 제거( 튀는 자료는 군집 제외 ) 등등


# 7. HDBCN 모델: 자동으로 최적의 K 하고 이상치 제거해주는 모델
# 외부라이브러리 설치 `py -m pip install hdbscan`
import hdbscan
# `min_cluster_size=` 최소 클러스터 개수
# `min_samples=` 클러스터들의 중심점이 되기 위한 최소 자료(샘플) 수 (최소 2개 이상부터 군집 가능)
# `prediction_data=True` 학습된 모델이 새로운 데이터를 예측할 경우 캐시(임시메모리) 활성화 (왠만하면 True)
hdb = hdbscan.HDBSCAN( min_cluster_size=2 , min_samples=2 , prediction_data=True )     # 객체 생성
hdb.fit( scaledDf )
print( hdb.labels_ )                                            # -1(이상치 샘플): 어디에도 속하지 않은 샘플
# [-1  2  2  0 -1  0 -1 -1 -1  3  4 -1  1  1  1 -1  2  2  1  1  2 -1  0  2  3  2 -1  2 -1 -1  3 -1  4  2  2  4  4 -1]
print( hdbscan.approximate_predict( hdb , scaledDf ) )
print( hdbscan.approximate_predict( hdb , scaledNewDf ) )   # [1] : 1그룹 예측