
from fastapi import APIRouter,Request,Response
router = APIRouter( prefix='/api/model')

from service import carService

# http://localhost:8000/api/model/admin
@router.post("/admin") # 매핑 주소 생성 
async def 학습요청( resquest : Request ):
    list = await resquest.json()
    print( resquest )
    return carService.학습요청(list)

# http://localhost:8000/api/model/user
@router.post("/user")
async def 예측요청( car : dict ) :
    print( car )
    return carService.예측요청(car)