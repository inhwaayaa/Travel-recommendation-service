# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 16:44:57 2024

@author: InQ
"""

from catboost import Pool
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import pandas as pd

# 특정 특성에 대한 Partial Dependence Plot
path = '.'

Train = pd.read_csv(path + '/source/관광지 추천시스템 Trainset_final_total.csv')

Train.drop(['TRAVELER_ID', 'REVISIT_INTENTION', 'RCMDTN_INTENTION','RESIDENCE_TIME_MIN',
            'REVISIT_YN','INCOME'], axis = 1, inplace = True)

Train['VISIT_AREA_TYPE_CD'] = Train['VISIT_AREA_TYPE_CD'].astype('string')

y_train = Train['DGSTFN']
X_train = Train.drop(['DGSTFN','TRAVEL_ID'], axis = 1)

# cat_features = ['VISIT_AREA_NM', 'SIDO', 'GUNGU', 'EUPMYEON', 'VISIT_AREA_TYPE_CD',
#             'TRAVEL_MISSION_PRIORITY', 'AGE_GRP', 'GENDER']

model = CatBoostRegressor(n_estimators = 400,
                          cat_features = ['VISIT_AREA_NM', 'SIDO', 'GUNGU', 'EUPMYEON', 'VISIT_AREA_TYPE_CD',
                                     'TRAVEL_MISSION_PRIORITY', 'AGE_GRP', 'GENDER'],
                          learning_rate = 0.02,
                          depth = 6,
                          random_state = 42)

# Pool 생성
Train['VISIT_AREA_NM'] = Train['VISIT_AREA_NM'].astype('string')
cat_feature_indices = [X_train.columns.get_loc('VISIT_AREA_NM')]

# Pool 생성
train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices)

# 모델 훈련
model.fit(train_pool)

# 특정 특성에 대한 Partial Dependence Plot
feature_of_interest = 'VISIT_AREA_NM'
plt.figure(figsize=(10, 6))
model.plot_partial_dependence(train_pool, features=[feature_of_interest], grid_resolution=50)
plt.show()
