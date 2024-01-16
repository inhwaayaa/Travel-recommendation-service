# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 12:38:17 2024

@author: InQ
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore') #warning 문구 제거
pd.set_option('display.max_columns', None)

'''
원천 데이터를 가지고 훈련데이터와 테스트데이터를 만듬

'''

#%%

#데이터 불러오기
path = '.'
visit_area_info = pd.read_csv(path + '/source/tn_visit_area_info_방문지정보_total.csv') # 방문지 정보 activity
travel = pd.read_csv(path+'/source/tn_travel_여행_total.csv') # 여행 travel
traveller_master = pd.read_csv(path+'/source/tn_traveller_master_여행객 Master_total.csv') #여행객 정보 Master traveler

#%%
# visit_area_info.drop(visit_area_info[visit_area_info['VISIT_AREA_NM'] == '용소폭포'].index, inplace=True)
# visit_area_info.drop(visit_area_info[visit_area_info['VISIT_AREA_NM'] == '송원'].index, inplace=True)
# visit_area_info.drop(visit_area_info[visit_area_info['VISIT_AREA_NM'] == '황금어장'].index, inplace=True)

#%%
a = visit_area_info[visit_area_info['VISIT_AREA_NM'] == '황금어장']
a[a['ROAD_NM_ADDR'].str.startswith('제주')].index                
visit_area_info.drop(a[a['ROAD_NM_ADDR'].str.startswith('제주')].index, inplace=True)
visit_area_info.to_csv(path + '/source/tn_visit_area_info_방문지정보_total2.csv')
# # 전처리

# ## 1) visit_area_info 방문지 정보 df

# In[4]:


# 관광지 선택
visit_area_info = visit_area_info[ (visit_area_info['VISIT_AREA_TYPE_CD'] == 1) |
                                  (visit_area_info['VISIT_AREA_TYPE_CD'] == 2) |
           (visit_area_info['VISIT_AREA_TYPE_CD'] == 3) | (visit_area_info['VISIT_AREA_TYPE_CD'] == 4) |
           (visit_area_info['VISIT_AREA_TYPE_CD'] == 5) | (visit_area_info['VISIT_AREA_TYPE_CD'] == 6) |
            (visit_area_info['VISIT_AREA_TYPE_CD'] == 7) | (visit_area_info['VISIT_AREA_TYPE_CD'] == 8)]



# In[7]:


visit_area_info.dropna(subset = ['LOTNO_ADDR'], inplace = True)
visit_area_info = visit_area_info.reset_index(drop = True)

for i in range(len(visit_area_info['LOTNO_ADDR'])):
    if len(visit_area_info['LOTNO_ADDR'][i].split(' '))<3:
        visit_area_info.drop(i, inplace=True)
visit_area_info = visit_area_info.reset_index(drop = True)
# In[8]:


# 시도/군구 변수 생성
sido = []
gungu = []
eupmyeon = []
for i in range(len(visit_area_info['LOTNO_ADDR'])):    
    sido.append(visit_area_info['LOTNO_ADDR'][i].split(' ')[0])
    gungu.append(visit_area_info['LOTNO_ADDR'][i].split(' ')[1])
    eupmyeon.append(visit_area_info['LOTNO_ADDR'][i].split(' ')[2])


# In[9]:


visit_area_info['SIDO'] = sido
visit_area_info['GUNGU'] = gungu
visit_area_info['EUPMYEON'] = eupmyeon



# In[10]:

## 추후 x,y 코드 가져와서 시각화 해야할 수 있다.
visit_area_info = visit_area_info[['TRAVEL_ID', 'VISIT_AREA_NM', 'SIDO', 'GUNGU', 'EUPMYEON', 'VISIT_AREA_TYPE_CD', 'DGSTFN',
                                  'REVISIT_INTENTION', 'RCMDTN_INTENTION', 'RESIDENCE_TIME_MIN', 'REVISIT_YN']]


# ## 2) travel 여행 정보 df

# In[11]:

# TRAVEL_MISSION_CHECK의 첫번째 항목 가져오기
travel_list = []
for i in range(len(travel)):
    value = int(travel['TRAVEL_MISSION_CHECK'][i].split(';')[0])
    travel_list.append(value)

travel['TRAVEL_MISSION_PRIORITY'] = travel_list

# In[12]:


travel = travel[['TRAVEL_ID', 'TRAVELER_ID', 'TRAVEL_MISSION_PRIORITY']]

# In[13]:


traveller_master = traveller_master[['TRAVELER_ID', 'GENDER', 'AGE_GRP', 'INCOME', 'TRAVEL_STYL_1', 
                                     'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4', 'TRAVEL_STYL_5', 
                                     'TRAVEL_STYL_6', 'TRAVEL_STYL_7','TRAVEL_STYL_8', 
                                      'TRAVEL_MOTIVE_1', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM' ]]


# ## 데이터 프레임 합치기

# In[14]:


df = pd.merge(travel, traveller_master, left_on = 'TRAVELER_ID', right_on = 'TRAVELER_ID', how = 'inner')


# In[15]:


df = pd.merge(visit_area_info, df, left_on = 'TRAVEL_ID', right_on = 'TRAVEL_ID', how = 'right')


# In[16]:


# ## 만족도(y) 결측치 삭제

# In[17]:


df = df.dropna(subset = ['DGSTFN'])
df.reset_index(drop=True, inplace=True)


# In[18]:


len(df['TRAVEL_ID'].unique())


# ## 체류시간 결측치 대체
# 

# 체류시간 0을 median 60으로 바꾸기

# In[19]:


df['RESIDENCE_TIME_MIN'] = df['RESIDENCE_TIME_MIN'].replace(0,60)


# ## 재방문여부 원핫인코딩

# In[20]:


df['REVISIT_YN'] = df['REVISIT_YN'].replace("N",0)
df['REVISIT_YN'] = df['REVISIT_YN'].replace("Y",1)


# ## 여행스타일 결측치 삭제

# In[21]:


df.dropna(subset = ['TRAVEL_STYL_1'], inplace = True)
df.reset_index(drop= True, inplace = True)


# In[25]: 시/도이름 통일
df['SIDO'].unique()
#%%
places = list(df['SIDO'])
for i in range(len(places)):
    if places[i][-1] == '도':
        if len(places[i])<=5:
            places[i] = places[i][:-1]
    elif places[i][-3:] == '광역시' or places[i][-3:] == '특별시':
        places[i] = places[i][:-3]
df['SIDO'] = places

df['SIDO'] = df['SIDO'].replace('경상북', '경북')
df['SIDO'] = df['SIDO'].replace('전라남', '전남')
df['SIDO'] = df['SIDO'].replace('경상남', '경남')
df['SIDO'] = df['SIDO'].replace('충청남', '충남')
df['SIDO'] = df['SIDO'].replace('전라북', '전북')
df['SIDO'] = df['SIDO'].replace('충청북', '충북')

df.drop(df[df['SIDO']=='광복동'].index, axis=0, inplace=True)
df.drop(df[df['SIDO']=='동부리'].index, axis=0, inplace=True)
df['SIDO'].unique()

# In[26]:

# Train세트와 test세트(df1)를 만듬
df1 = df
Train = pd.DataFrame(columns = list(df.columns))
for i in tqdm(list(df['VISIT_AREA_NM'].unique())): # 유니크한 관광지 목록 중에서
    df2 = df1[df1['VISIT_AREA_NM'] == i] # 특정 관광지에 간 모든 사람 뽑아서
    np.random.seed(42)
    if df2.empty:
        pass
    else:
        random_number = np.random.randint(len(df2)) 
        df_id = df2.iloc[[random_number]] # 그 중 랜덤으로 관광지에 간 사람 한 명 뽑아서
        index = df_id.iloc[0,0]
        df3 = df1[df1['TRAVEL_ID'] == index] #그 사람이 간 모든 관광지를 구해서
        df1 = pd.merge(df3, df1, how = 'outer', indicator = True)
        df1 = df1.query('_merge =="right_only"').drop(columns = ['_merge']) # 기존 데이터프레임에서 그 사람 내용을 삭제하고
        Train = pd.concat([Train,df3], ignore_index=True) #train set 에 추가


# In[27]:

# train 0.8 // test 0.2 비율로 맞추기
while len(df1)/len(df) > 0.2:
    np.random.seed(42)
    random_number = np.random.randint(len(df1))
    df_id = df1.iloc[[random_number]]
    index = df_id.iloc[0,0]
    df3 = df1[df1['TRAVEL_ID'] == index]
    df1 = pd.merge(df3, df1, how = 'outer', indicator = True)
    df1 = df1.query('_merge =="right_only"').drop(columns = ['_merge'])
    Train = pd.concat([Train, df3], ignore_index=True)


# ## Train set에서 방문지에 대한 변수 생성
# 방문지마다 체류시간 평균, 추천의향의 평균, 재방문여부의 평균, 동반자 수의 평균, 재방문의향의 평균 산출

# In[29]:


#새로운 데이터프레임 생성해서, 이 데이터프레임에 평균값을 추가한 새로운 Train set 생성할 것임

new_train = pd.DataFrame(columns = list(Train.columns) + ['RESIDENCE_TIME_MIN_mean', 'RCMDTN_INTENTION_mean',
                                                          'REVISIT_YN_mean', 'TRAVEL_COMPANIONS_NUM_mean',
                                                          'REVISIT_INTENTION_mean'])

# globals()[str(j)+'_mean'] 이거 그냥 전역변수 이렇게 생각하지 말고 변수를 동적으로 생성한다고 생각해야함
# 쉽게 생각하려면 그냥 첫 j에서 문자열 'RESIDENCE_TIME_MIN_mean'과 똑같다.

# 기존 열들로 새로운 열 만드는 반복문 new_train에 새로운 훈련셋이 담김
for i in tqdm(list(Train['VISIT_AREA_NM'].unique())): #유니크한 관광지 목록 중에서
    df2 = Train[Train['VISIT_AREA_NM'] == i] # 특정 관광지에 간 모든 사람 뽑아서
    for j in ['RESIDENCE_TIME_MIN', 'RCMDTN_INTENTION', 'REVISIT_YN', 'TRAVEL_COMPANIONS_NUM', 'REVISIT_INTENTION']:
        #체류시간 평균 산출 
        globals()[str(j)+'_mean'] = df2[str(j)]
        globals()[str(j)+'_mean'] = np.mean(globals()[str(j)+'_mean'])
        #데이터프레임에 들어가게 값을 리스트 형태로 변환
        globals()[str(j)+'_mean'] = np.repeat(globals()[str(j)+'_mean'], len(df2)) 
        df2[str(j)+'_mean'] = globals()[str(j)+'_mean']
    #새로운 데이터프레임에 방문지별 평균값 대입
    new_train = pd.concat([new_train, df2], axis = 0)


# In[30]:


#편의를 위해 유저별 정렬
new_train.sort_values(by = ['TRAVEL_ID'], axis = 0, inplace = True)


# ## DATA SET 저장

# In[31]:


#train set 저장
new_train.to_csv(path + '/source/관광지 추천시스템 Trainset_final.csv', index = False)
#test set 저장
df1.to_csv(path + '/source/관광지 추천시스템 Testset_final.csv', index = False)

