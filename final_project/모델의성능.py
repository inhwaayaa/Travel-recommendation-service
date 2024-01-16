# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 15:31:46 2024

@author: ham90
"""
import pandas as pd
import warnings
from catboost import CatBoostRegressor
import numpy as np
import joblib
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore') #warning 문구 제거
pd.set_option('display.max_columns', None)
#%%
path = '.'

Train = pd.read_csv(path + '/source/관광지 추천시스템 Trainset_final_total.csv')

Train.drop(['TRAVELER_ID', 'REVISIT_INTENTION', 'RCMDTN_INTENTION','RESIDENCE_TIME_MIN',
            'REVISIT_YN','INCOME'], axis = 1, inplace = True)

Train['VISIT_AREA_TYPE_CD'] = Train['VISIT_AREA_TYPE_CD'].astype('string')

y_train = Train['DGSTFN']
X_train = Train.drop(['DGSTFN','TRAVEL_ID'], axis = 1)



# In[56]:

model = joblib.load(path + '/ML/bestCatboost_model_D_travel.pkl')
#%%
new_test = pd.read_csv(path + '/source/단순예측 Testset_final_total.csv')
new_test['VISIT_AREA_TYPE_CD'] = new_test['VISIT_AREA_TYPE_CD'].astype('string')
#%%
y_test = new_test['DGSTFN']
X_test = new_test.drop(['DGSTFN'], axis = 1)
#%%
# 모델의 예측값과 실제 레이블 가져오기
y_pred = model.predict(X_test)
y_true = y_test  # 실제 레이블

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true, y_pred)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)

r_sq = model.score(X_train, y_train)
r_sq2 = model.score(X_test, y_test)

print('평균제곱오차: ', mse)
print('평균절대오차: ', mae)
print('결정계수_train: ', r_sq)
print('결정계수_test: ', r_sq2)

feature_importance = model.get_feature_importance()

# 특성 중요도 시각화
sorted_idx = np.argsort(feature_importance)
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), X_train.columns[sorted_idx])
plt.xlabel('Feature Importance')
plt.show()