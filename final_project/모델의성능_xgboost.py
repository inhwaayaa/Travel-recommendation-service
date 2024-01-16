# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 15:31:46 2024

@author: ham90
"""
import pandas as pd
import warnings
import xgboost
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import explained_variance_score
warnings.filterwarnings('ignore') #warning 문구 제거
pd.set_option('display.max_columns', None)

path = '.'

Train = pd.read_csv(path + '/source/관광지 추천시스템 Trainset_final_total.csv')

Train.drop(['TRAVELER_ID', 'REVISIT_INTENTION', 'RCMDTN_INTENTION','RESIDENCE_TIME_MIN',
            'REVISIT_YN','INCOME'], axis = 1, inplace = True)

#Train['VISIT_AREA_TYPE_CD'] = Train['VISIT_AREA_TYPE_CD'].astype('string')

y_train = Train['DGSTFN']
X_train = Train.drop(['DGSTFN','TRAVEL_ID'], axis = 1)

new_test = pd.read_csv(path + '/source/단순예측 Testset_final_total.csv')
#%%
y_test = new_test['DGSTFN']
X_test = new_test.drop(['DGSTFN'], axis = 1)
# ... (이전 코드 부분은 그대로 유지)

# Train 데이터셋 변환
categorical_cols = ['VISIT_AREA_NM', 'SIDO', 'GUNGU', 'EUPMYEON', 'VISIT_AREA_TYPE_CD', 'GENDER']
X_train[categorical_cols] = X_train[categorical_cols].astype('category')

# Test 데이터셋 변환
X_test[categorical_cols] = X_test[categorical_cols].astype('category')

X_test = X_test[X_train.columns]

model = joblib.load(path + '/ML/bestXgboost_model_D_travel.pkl')

# 모델의 예측값과 실제 레이블 가져오기
y_pred = model.predict(X_test)
y_true = y_test  # 실제 레이블

# 평가 지표 계산
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
#%%
# 결과 출력
print(f'평균제곱오차: {mse}')
print(f'평균절대오차: {mae}')

# 특성 중요도 시각화
xgboost.plot_importance(model)
plt.show()
r_sq = model.score(X_train, y_train)
r_sq2 = model.score(X_test, y_test)
print('결정계수_train: ', r_sq)
print('결정계수_test: ', r_sq2)

