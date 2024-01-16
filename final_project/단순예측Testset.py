# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 09:19:33 2024

@author: InQ
"""
import pandas as pd
from tqdm import tqdm
import numpy as np
#%%
path = '.'
test = pd.read_csv(path + '/source/관광지 추천시스템 Testset_final_total.csv')

new_test = pd.DataFrame(columns = list(test.columns) + ['RESIDENCE_TIME_MIN_mean', 'RCMDTN_INTENTION_mean',
                                                          'REVISIT_YN_mean', 'TRAVEL_COMPANIONS_NUM_mean',
                                                          'REVISIT_INTENTION_mean'])

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

#편의를 위해 유저별 정렬
new_test.sort_values(by = ['TRAVEL_ID'], axis = 0, inplace = True)

new_test.drop(['TRAVEL_ID','TRAVELER_ID', 'REVISIT_INTENTION', 'RCMDTN_INTENTION','RESIDENCE_TIME_MIN',
            'REVISIT_YN','INCOME'], axis = 1, inplace = True)
#%%
new_test.to_csv(path + '/source/단순예측 Testset_final_total.csv', index=False)