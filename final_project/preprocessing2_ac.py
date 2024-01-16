# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 10:11:05 2024

@author: ham90
"""

import pandas as pd
from tqdm import tqdm
import warnings

'''
preprocessing1을 통해 얻은 훈련 데이터, 테스트 데이터를 좀 더 가공함
'''

warnings.filterwarnings('ignore') #warning 문구 제거
pd.set_option('display.max_columns', None)

path ='.'

traind = pd.read_csv(path + '/source/관광지 추천시스템 Trainset_final_ac.csv')
testd = pd.read_csv(path + '/source/관광지 추천시스템 Testset_final_ac.csv')

y_testd = testd['DGSTFN']
X_testd = testd.drop(['DGSTFN'], axis = 1)


#%% 2회 이상 관광한 방문지 리스트 만들기
info = traind[['SIDO', 'VISIT_AREA_NM', 'GUNGU', 'EUPMYEON', 'VISIT_AREA_TYPE_CD', 'RESIDENCE_TIME_MIN_mean', 'RCMDTN_INTENTION_mean',
'REVISIT_YN_mean', 'TRAVEL_COMPANIONS_NUM_mean', 'REVISIT_INTENTION_mean']]
info.drop_duplicates(['VISIT_AREA_NM'], inplace = True)

visiting_list = traind[['VISIT_AREA_NM']] #train set에 있는 방문지에 대해서만 2회 이상 방문하였는지 확인
visiting_list.reset_index(drop = True, inplace = True)

dfdf = pd.DataFrame(visiting_list.value_counts(), columns = ['count'])
dfdf['VISIT_AREA_NM'] = dfdf.index
dfdf.reset_index(drop = True, inplace = True)

for i in range(len(dfdf)):
    dfdf['VISIT_AREA_NM'][i] = str(dfdf['VISIT_AREA_NM'][i])
    dfdf['VISIT_AREA_NM'][i] = dfdf['VISIT_AREA_NM'][i].replace("(","").replace(")","").replace(",","").replace("\''","")
    dfdf['VISIT_AREA_NM'][i] = dfdf['VISIT_AREA_NM'][i][1:-1]

dfdf = dfdf[dfdf['count'] >= 2]

visit_list = list(dfdf['VISIT_AREA_NM']) #visit_list에 2회 이상 방문지 리스트

#%%
#방문지가 2회 이상 방문한 관광지 아니면 제거

info.reset_index(drop = True, inplace = True)

for i in tqdm(range(len(info))):
    if info['VISIT_AREA_NM'][i] not in visit_list:
        info = info.drop([i], axis = 0)
info.reset_index(drop = True, inplace = True)
#%% info와 dfdf가 왜 다른지 알아보기 위한 코드 위에 ()를 빼는 과정에서 안에 있는 ()까지 빼내버려서 달라짐
# info_list = list(info['VISIT_AREA_NM'])
# for i in range(len(dfdf)):
#     if dfdf['VISIT_AREA_NM'][i] not in info_list:
#         print(dfdf['VISIT_AREA_NM'][i])

# In[68]:


#여행지 정보 저장
info.reset_index(drop = True, inplace = True)

#%% 시/도 이름 통일

#%%
#print(info[info['VISIT_AREA_NM'].str.endswith('숙소')])
info = info[~info['VISIT_AREA_NM'].str.endswith('집')]
info = info[~info['VISIT_AREA_NM'].str.endswith('숙소')]


#%% 
info.to_csv(path + '/source/관광지 추천시스템 여행지 정보 방문 2회 이상_ac.csv', index=False)
