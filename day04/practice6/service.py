import pandas as pd
import httpx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge , Lasso

class CarService:
    def __init__(self):
        self.model = None

    # 1. 
    def 학습요청( self , carList ):
        df = pd.DataFrame( carList )
        trian_data = df[['평균연비', '누적주행거리키로', '출고후경과월수', '사고감가건수', '소유자변경횟수']]
        target_data = df['매매가격만원'].values
        train_input , test_input , train_target , test_target = train_test_split( trian_data , target_data , test_size=0.2, random_state=42)
        lr = LinearRegression()
        lr.fit( train_input , train_target )
        print( lr.score( test_input , test_target ) )
        self.model = lr
        return True
    
    # 2. 
    def 예측요청( self , car ):
        if self.model is None:
            return "[ 학습모델 부재 ]"
        del car['차량번호ID']
        del car['매매가격만원']
        car_features = [ car['평균연비'], car['누적주행거리키로'], car['출고후경과월수'],  car['사고감가건수'],  car['소유자변경횟수'] ]
        predict = self.model.predict([car_features])
        return predict[0]
    
carService = CarService()