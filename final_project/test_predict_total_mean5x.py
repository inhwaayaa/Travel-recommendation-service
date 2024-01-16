# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 10:11:05 2024

@author: ham90
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import joblib
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore') #warning 문구 제거
pd.set_option('display.max_columns', None)

path ='.'

Train = pd.read_csv(path + '/source/관광지 추천시스템 Trainset_final_total.csv')

Train.drop(['TRAVELER_ID', 'REVISIT_INTENTION', 'RCMDTN_INTENTION','RESIDENCE_TIME_MIN',
            'REVISIT_YN','INCOME','RESIDENCE_TIME_MIN_mean', 'RCMDTN_INTENTION_mean',
            'REVISIT_YN_mean', 'TRAVEL_COMPANIONS_NUM_mean', 'REVISIT_INTENTION_mean'], axis = 1, inplace = True)

Train['VISIT_AREA_TYPE_CD'] = Train['VISIT_AREA_TYPE_CD'].astype('string')

y_train = Train['DGSTFN']
X_train = Train.drop(['DGSTFN','TRAVEL_ID'], axis = 1)



# In[56]:


model = CatBoostRegressor(n_estimators = 400,
                          cat_features = ['VISIT_AREA_NM', 'SIDO', 'GUNGU', 'EUPMYEON', 'VISIT_AREA_TYPE_CD',
                                      'TRAVEL_MISSION_PRIORITY', 'AGE_GRP', 'GENDER'],
                          learning_rate = 0.02,
                          depth = 6,
                          random_state = 42)

    # 훈련시키고
model.fit(X_train, y_train, early_stopping_rounds=5)


# In[57]:


now = time
print(now.strftime('%Y-%m-%d %H:%M:%S'))


# ## 모델 저장

# In[58]:


joblib.dump(model,path + '/ML/catboost_model_D_travel_total.pkl')


# In[59]:


modeld = joblib.load(path + '/ML/catboost_model_D_travel_total.pkl')
traind = pd.read_csv(path + '/source/관광지 추천시스템 Trainset_final_total.csv')
testd = pd.read_csv(path + '/source/관광지 추천시스템 Testset_final_total.csv')

y_testd = testd['DGSTFN']
X_testd = testd.drop(['DGSTFN'], axis = 1)


# In[62]:


#유저정보
data = testd[['TRAVEL_ID', 'SIDO', 'GUNGU', 'EUPMYEON', 'TRAVEL_MISSION_PRIORITY', 'GENDER', 'AGE_GRP',
                            'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
                            'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
                            'TRAVEL_MOTIVE_1', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM']]


# In[63]:


#내가 간 시도 군구 리스트:
data1 = pd.DataFrame(columns=['TRAVEL_ID', 'TRAVEL_MISSION_PRIORITY', 'GENDER', 'AGE_GRP',
                            'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
                            'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
                            'TRAVEL_MOTIVE_1', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM', 'sido_gungu_list'])
for i in tqdm(list(data['TRAVEL_ID'].unique())):
    temp_df = data[data['TRAVEL_ID'] == i]
    temp_df1 = temp_df[['SIDO', 'GUNGU', 'EUPMYEON']] #각 유저별 방문한 시군구 확인
    temp_df1.reset_index(drop = True, inplace = True)
    sido_gungu_visit = []
    for j in range(len(temp_df1)):
        sido_gungu_visit.append(temp_df1['SIDO'][j] + '+' + temp_df1['GUNGU'][j] + '+' + temp_df1['EUPMYEON'][j])
    sido_gungu_list = list(set(sido_gungu_visit))
    new = temp_df.drop(['SIDO', 'GUNGU', 'EUPMYEON'], axis = 1) #기존 시도, 군구 제외하고
    new = new.head(1)
    new['sido_gungu_list'] = str(sido_gungu_list)
    data1 = pd.concat([data1, new], axis = 0) #새로운 데이터프레임 생성        
    


# In[64]:


#유저 정보 저장
data1.reset_index(drop = True, inplace = True)
data1.to_csv(path + '/source/관광지 추천시스템 Testset_D-1 유저 정보_total.csv', index=False)


# In[70]:

# 유저 정보
data = pd.read_csv(path + '/source/관광지 추천시스템 Testset_D-1 유저 정보_total.csv')
# 여행지 정보
info = pd.read_csv(path + '/source/관광지 추천시스템 여행지 정보 방문 2회 이상_total.csv')

# In[71]:


result = []
for i in tqdm(range(len(data1))):
    #데이터
    
    final_df = pd.DataFrame(columns = ['VISIT_AREA_NM', 'SIDO', 'GUNGU', 'EUPMYEON', 'VISIT_AREA_TYPE_CD',
           'TRAVEL_MISSION_PRIORITY', 'GENDER', 'AGE_GRP',
           'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
           'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
           'TRAVEL_MOTIVE_1', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM']) #빈 데이터프레임에 내용 추가
    ####### 시/도 군/구 별 자료 수집
    temp = data1['sido_gungu_list'][i].replace("[","").replace("]","").replace("\'","").replace(", ",",")
    places_list = list(map(str, temp.split(",")))
    for q in places_list:
        sido, gungu, eupmyeon = map(str, q.split("+"))
        info_df = info[(info['SIDO'] == sido) & (info['GUNGU'] == gungu) & (info['EUPMYEON'] == eupmyeon)] 
               
        info_df.reset_index(inplace = True, drop = True)
        data2 = data1.drop(['sido_gungu_list','TRAVEL_ID'], axis =1)
        user_df = pd.DataFrame([data2.iloc[i].to_list()]*len(info_df), columns = ['TRAVEL_MISSION_PRIORITY', 'GENDER', 'AGE_GRP',
                            'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
                            'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
                            'TRAVEL_MOTIVE_1', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM'])
        df = pd.concat([user_df, info_df], axis = 1)
        df = df[['VISIT_AREA_NM', 'SIDO', 'GUNGU', 'EUPMYEON', 'VISIT_AREA_TYPE_CD',
       'TRAVEL_MISSION_PRIORITY', 'GENDER', 'AGE_GRP',
       'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
       'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
       'TRAVEL_MOTIVE_1', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM']] # 변수정렬
        df['VISIT_AREA_TYPE_CD'] = df['VISIT_AREA_TYPE_CD'].astype('string')
        final_df = pd.concat([final_df, df], axis = 0)
    final_df.reset_index(drop = True, inplace = True)
    final_df.drop_duplicates(['VISIT_AREA_NM'], inplace = True)

    #모델 예측
    y_pred = modeld.predict(final_df)
    y_pred = pd.DataFrame(y_pred, columns = ['y_pred'])
    test_df1 = pd.concat([final_df, y_pred], axis = 1)
    test_df1.sort_values(by = ['y_pred'], axis = 0, ascending=False, inplace = True) # 예측치가 높은 순대로 정렬

    test_df1 = test_df1.iloc[0:10,] #상위 10개 관광지 추천

    visiting_candidates = list(test_df1['VISIT_AREA_NM']) # 모델이 추천한 관광지들을 리스트 형태로 변환

# 유저정보와 추천 관광지
    test_df2 = test_df1[['TRAVEL_MISSION_PRIORITY', 'GENDER', 'AGE_GRP',
                        'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
                        'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
                        'TRAVEL_MOTIVE_1', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM']]
    if len(test_df2) == 0:
        rec = []
        result.append(rec)        
   
    else:
        
        rec = test_df2.iloc[0].to_list()

        rec.append(visiting_candidates)

        result.append(rec)



# In[72]:

final_df = pd.DataFrame(result,
                            columns = ['TRAVEL_MISSION_PRIORITY', 'GENDER', 'AGE_GRP',
                            'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
                            'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
                            'TRAVEL_MOTIVE_1', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM', 'recommend_result_place'])
final_df = final_df[['recommend_result_place']]
travel_id = data1[['TRAVEL_ID']]
travel_id.reset_index(drop = True, inplace = True)
final_df = pd.concat([travel_id, final_df], axis = 1)


# In[73]:


#output_df 저장
final_df.to_csv(path + '/source/관광지 추천시스템 Testset- OUTPUT_D1_total.csv', index=False)
#output_df 불러오기
final_df = pd.read_csv(path + '/source/관광지 추천시스템 Testset- OUTPUT_D1_total.csv')


# ## 실제 유저가 다녀온 관광지랑 비교하여 Recall@10 산출

# In[74]:


final_df
# In[75]:


#추천지 10개 미만인 여행 ID확인
#이 숫자를 줄여도 성능이 올라가지 않을까?
travel_id_list = []
for i in tqdm(range(len(final_df))):
    recommend_list = final_df['recommend_result_place'][i]
    if pd.isna(recommend_list):
    # if not recommend_list:
        travel_id_list.append(final_df['TRAVEL_ID'][i])
        continue
    if recommend_list.count(',') < 9:
        travel_id_list.append(final_df['TRAVEL_ID'][i])
        


# In[76]:


#군/군 리스트 출력
## 이 부분 해결 하기 위해서 군/구 목록 places라는 변수에 뽑아오기
'''
4. 성능을 올리기 위해 다음과 같은 추가 방법 도입:
- 아무래도 사람이 수기로 데이터를 입력하다 보니까 유사/동일 장소를 갔음에도 컴퓨터는 다른 장소로 인식 
    -ex1: ’파라다이스시티‘와 ’파라다이스시티 주차장‘을 다른 장소로 인식
    -ex2: ‘국립중앙박물관 특별전시관’과 ‘국립중앙박물관’을 다른 장소로 인식
- 그러므로 ‘모델이 추천한 추천지 10개’와 ‘유저가 만족한다고 했던 곳’을 단어 별로 쪼개서 공통어가 있으면 교집합 개수에 추가
    -ex1: ‘파라다이스시티’와 ‘파라다이스시티 주차장’은 ‘파라다이스시티’라는 공통어가 있으므로 교집합 개수에 추가
    -ex2: ‘국립중앙박물관 특별전시관’과 ‘국립중앙박물관’은 ‘국립중앙박물관’이라는 공통어가 있으므로 교집합 개수에 추가
- ‘파주점’, ‘하남점’ 등 지점명이 공통어가 되는 경우 배제
    -ex1: ‘롯데프리미엄아울렛 파주점'이라는 장소가 있을 때, 마지막 단어의 마지막 글자가 '점'일 경우에 마지막 글자를 제거 (공통어 비교 시 '롯데프리미엄아울렛'만 비교)
- 유저가 방문한 장소에 군/구가 공통어가 되는 경우 배제
    -ex1: 유저가 방문한 '스타필드 고양'은 ’스타필드‘, ’고양‘으로 나누어지는데, '고양'이라는 군/구 명을 제거해서 ’고양 어울림누리‘ 같은 장소와 교차어로 포함되지 않도록 함
'''
places = list(set(X_testd['GUNGU']))
for i in range(len(places)):
    if places[i][-1] == '구' or places[i][-1] == '시' or places[i][-1] == '군':
        places[i] = places[i][:-1]


# In[77]:


#유저가 다녀온 관광지 중에서 만족도가 4이상인 관광지 목록
recall_10_list = []
visit_list = list(info['VISIT_AREA_NM'])
for i in tqdm(list(testd['TRAVEL_ID'].unique())):
    
    #추천한 방문지가 10개 미만이면 0
    if i in travel_id_list:
        recall_10_list.append(0)
        continue
    
    
    
    satisfied = testd[testd['TRAVEL_ID'] == i] #실제(y_actual) 관광객이 만족한 관광지
    satisfied.reset_index(drop = True, inplace = True) 
    

    satisfied1 = satisfied[satisfied['DGSTFN'] >=4 ] #만족의 기준은 4이상 일때만 만족이라고 정의
    if len(satisfied1) == 0: # 유저가 만족한 관광지가 하나도 없으면 recall@10은 어차피 0
        recall_10_list.append(0)
        continue
    else:
        item_list = satisfied1['VISIT_AREA_NM']
                
                
    item_list = list(set(item_list))
    
#final_df의 추천지 10개랑 비교
    recommend_list = final_df[final_df['TRAVEL_ID'] == i]['recommend_result_place'] #모델 추천 관광지 30개

    # #추천한 방문지가 10개 미만이면
    # summ0 = 0
    # if i in travel_id_list:
    #     #recall_10_list.append(0)
    #     for n in item_list:
    #         word_list = list(n.split(' '))
    #         if word_list[-1][-1] == '점': #지점명 삭제
    #             del word_list[-1]
    #         for o in word_list:
    #             if o in places:#장소에 군/구 명 있으면 아무것도 하지 않고 스킵
    #                 pass
    #             else:
    #                 for p in recommend_list: #장소에 교차어 있으면 해당 장소는 방문했다고 인식하기
    #                     if o in str(p) :
    #                         summ0 += 1
    #     recall10_for_1user = summ0 / min(10, len(satisfied1)) #recall@10 산식
    #     if recall10_for_1user > 1:
    #         recall10_for_1user = 1
    #     recall10_for_1user = recall10_for_1user*0.5
    #     recall_10_list.append(recall10_for_1user)
    #     continue
    
    summ = 0
    for n in item_list:
        word_list = list(n.split(' '))
        if word_list[-1][-1] == '점': #지점명 삭제
            del word_list[-1]
        for o in word_list:
            if o in places:#장소에 군/구 명 있으면 아무것도 하지 않고 스킵
                pass
            else:
                for p in recommend_list: #장소에 교차어 있으면 해당 장소는 방문했다고 인식하기
                    if o in str(p) :
                        summ += 1
    recall10_for_1user = summ / min(10, len(satisfied1)) #recall@10 산식
    if recall10_for_1user > 1:
        recall10_for_1user = 1
    recall_10_list.append(recall10_for_1user)


# In[78]:

now = time
print(now.strftime('%Y-%m-%d %H:%M:%S'))


# In[79]:


#recall@10 구하기 

recall_10 = np.mean(recall_10_list)
#sum(recall_10_list) / len(recall_10_list)

# In[80]:


print('최종성능:', recall_10)
