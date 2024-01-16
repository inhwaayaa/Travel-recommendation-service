# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 16:23:50 2024

@author: InQ
"""
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
# 모델 생성 및 학습
path='.'
model = joblib.load(path + '/ML/catboost_model_D_travel_total.pkl')

Train = pd.read_csv(path + '/source/관광지 추천시스템 Trainset_final_total.csv')

Train.drop(['TRAVELER_ID', 'REVISIT_INTENTION', 'RCMDTN_INTENTION','RESIDENCE_TIME_MIN',
            'REVISIT_YN','INCOME'], axis = 1, inplace = True)

Train['VISIT_AREA_TYPE_CD'] = Train['VISIT_AREA_TYPE_CD'].astype('string')

y_train = Train['DGSTFN']
X_train = Train.drop(['DGSTFN','TRAVEL_ID'], axis = 1)

#%% Feature Importance Plot(각 특성의 중요도)

feature_importance = model.get_feature_importance()
feature_names = X_train.columns
sorted_idx = feature_importance.argsort()

plt.barh(range(len(feature_names)), feature_importance[sorted_idx])
plt.yticks(range(len(feature_names)), feature_names[sorted_idx])
plt.xlabel('Feature Importance')
plt.show()

#%% SHAP (SHapley Additive exPlanations) Values

import shap

test = pd.read_csv(path + '/source/관광지 추천시스템 Testset_final_total.csv')

new_test = pd.DataFrame(columns = list(test.columns) + ['RESIDENCE_TIME_MIN_mean', 'RCMDTN_INTENTION_mean',
                                                          'REVISIT_YN_mean', 'TRAVEL_COMPANIONS_NUM_mean',
                                                          'REVISIT_INTENTION_mean'])


# 기존 열들로 새로운 열 만드는 반복문 new_train에 새로운 훈련셋이 담김
for i in tqdm(list(test['VISIT_AREA_NM'].unique())): #유니크한 관광지 목록 중에서
    df2 = test[test['VISIT_AREA_NM'] == i] # 특정 관광지에 간 모든 사람 뽑아서
    for j in ['RESIDENCE_TIME_MIN', 'RCMDTN_INTENTION', 'REVISIT_YN', 'TRAVEL_COMPANIONS_NUM', 'REVISIT_INTENTION']:
        #체류시간 평균 산출 
        globals()[str(j)+'_mean'] = df2[str(j)]
        globals()[str(j)+'_mean'] = np.mean(globals()[str(j)+'_mean'])
        #데이터프레임에 들어가게 값을 리스트 형태로 변환
        globals()[str(j)+'_mean'] = np.repeat(globals()[str(j)+'_mean'], len(df2)) 
        df2[str(j)+'_mean'] = globals()[str(j)+'_mean']
    #새로운 데이터프레임에 방문지별 평균값 대입
    new_test = pd.concat([new_test, df2], axis = 0)


new_test.drop(['TRAVEL_ID','TRAVELER_ID', 'REVISIT_INTENTION', 'RCMDTN_INTENTION','RESIDENCE_TIME_MIN',
            'REVISIT_YN','INCOME'], axis = 1, inplace = True)
new_test['VISIT_AREA_TYPE_CD'] = test['VISIT_AREA_TYPE_CD'].astype('string')


y_test = new_test['DGSTFN']
X_test = new_test.drop(['DGSTFN'], axis = 1)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 특정 샘플에 대한 SHAP Summary Plot
shap.summary_plot(shap_values, X_test)
