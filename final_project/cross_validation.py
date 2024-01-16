# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 16:45:10 2024

@author: InQ
"""
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore') #warning 문구 제거
pd.set_option('display.max_columns', None)
#%%
path = '.'

Train = pd.read_csv(path + '/source/관광지 추천시스템 Trainset_final_total.csv')
Train.drop(['TRAVELER_ID', 'REVISIT_INTENTION', 'RCMDTN_INTENTION','RESIDENCE_TIME_MIN',
            'REVISIT_YN','INCOME'], axis = 1, inplace = True)

Train['VISIT_AREA_TYPE_CD'] = Train['VISIT_AREA_TYPE_CD'].astype('string')


# In[52]:

y_train = Train['DGSTFN']
X_train = Train.drop(['DGSTFN'], axis = 1)

#%%
cv = 5
random_state = 42


X_train1 = X_train #변수 옮기기
y_train1 = y_train #변수 옮기기

# cv수 만큼 훈련세트, 훈련세트타겟 생성
for i in range(cv): #각 fold마다의 X_train, y_train 생성 (Train_1, Train_2, ... / target_1, target_2, ...)
    globals()['Train_'+str(i+1)] = pd.DataFrame(columns = list(X_train.columns))
    globals()['target_'+str(i+1)] = []
print(str(cv)+'개의 fold를 생성중입니다.....')
for i in tqdm(range(cv)):
    np.random.seed(random_state) #초기 시드 설정
    while (len(globals()['Train_'+str(i+1)]) / len(X_train)) < (1/cv): #1/cv 비율 만큼의 데이터가 모일 때까지
        random_number = np.random.randint(len(X_train1))
        df_id = X_train1.iloc[[random_number]]
        index = df_id.iloc[0,0] #랜덤하게 유저를 선택하고
        df1 = X_train1[X_train1['TRAVEL_ID'] == index] #그 유저가 갔던 모든 여행지 불러오고
        target_index = X_train1[X_train1['TRAVEL_ID'] == index].index
        X_train1 = pd.merge(df1, X_train1, how = 'outer', indicator = True)
        X_train1 = X_train1.query('_merge =="right_only"').drop(columns = ['_merge']) #기존 데이터프레임은 해당 유저 정보 삭제
        globals()['Train_'+str(i+1)] = pd.concat([globals()['Train_'+str(i+1)], df1], ignore_index=True) #validation set에 유저의 X_train 삽입
        globals()['target_'+str(i+1)].extend(list(y_train[list(target_index)])) #유저의 X_train에 상응하는 y_train 삽입
        if len(X_train1) == 0: #기존 데이터프레임에 모든 유저정보가 사라지면 validation set 생성이 완료된 것이므로 정지
            break
print(str(cv)+'개의 fold 생성이 완료되었습니다!')

#%%
np.random.seed(random_state)
initial = 0
number = 2
learning_rate = 0.02
depth = 6
early_stopping_rounds = 5
n_estimators = 400
    
final_recall = [] # K개의 검증 성능이 들어갈 리스트

for j in range(cv): #K개 fold중
    # cv수 만큼 fold를 나누고 그 중 하나는 검증세트로, 나머지는 합쳐서 훈련세트로 만듬
    # y_new_train에는 타겟값
    #한 fold에 대해서 학습
    combine_df_list = list(range(1, (cv+1))) # 1부터 cv까지 숫자리스트 만들어서
    del combine_df_list[j] #숫자 하나를 지우고, 그 숫자가 있는 Train set을 Validation set으로 설정
    #예를 들어 1이 빠졌으면 Train_1이 validation set, Train_2, Train_3, ... 는 Train set
    X_new_train = pd.DataFrame(columns = list(globals()['Train_'+str(j+1)].columns))
    y_new_train = []
    for i in combine_df_list: #지운 숫자 외의 숫자가 있는 Train set들을 결합
        X_new_train = pd.concat([X_new_train, globals()['Train_'+str(i)]], axis = 0) #X_train 결합
        y_new_train.extend(globals()['target_'+str(i)]) #y_train 결합
    y_new_train = np.array(y_new_train).astype(float)
    X_new_train.drop(['TRAVEL_ID'], axis = 1, inplace = True) #필요 없는 컬럼 제거
    if 'DGSTFN' in list(X_new_train.columns): #global 함수에서 발생하는 오류 해결
        X_new_train.drop(['DGSTFN'], axis = 1, inplace = True)
    # 캣부스트회귀 모델 만들고
    model = CatBoostRegressor(n_estimators = n_estimators,
                          cat_features = ['VISIT_AREA_NM', 'SIDO', 'GUNGU', 'EUPMYEON', 'VISIT_AREA_TYPE_CD',
                                      'TRAVEL_MISSION_PRIORITY', 'AGE_GRP', 'GENDER'],
                          learning_rate = learning_rate,
                          depth = depth,
                          random_state = 42)
##############################################################################################
    # 훈련시키고
    model.fit(X_new_train, y_new_train, early_stopping_rounds=early_stopping_rounds) #########모델 적합

    #학습한 fold에 대한 test 값 도출
    # 여기에 결국 검증세트로 predict했을때 얼만큼의 성능이 나왔는지에 대한 지표가 들어갈거임
    # 자세한건 밑에서 다시 나올때
    recall_10_list = [] #validation set의 recall 측정값들이 들어갈 리스트
    #####################유저 정보##################################
    # 유저 정보는 검증세트에서 가저온다
    # 결국 여기서 얻는건 data1이라는 데이터프레임인데 data1은 웹에서 얻게되는 이용자의 유저 정보로 보면됨
    # 근데 이제 sido_gungu_list열이 추가가 되는데 이건 이용자가 원하는 여행장소(시군구)로 볼 수 있다.
    # data1은 중복id없음
    data = globals()['Train_'+str(j+1)][['TRAVEL_ID', 'SIDO', 'GUNGU', 'EUPMYEON', 'TRAVEL_MISSION_PRIORITY', 'GENDER', 'AGE_GRP',
                            'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
                            'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
                            'TRAVEL_MOTIVE_1', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM']]
    data1 = pd.DataFrame(columns=['TRAVEL_ID', 'TRAVEL_MISSION_PRIORITY', 'GENDER', 'AGE_GRP',
                            'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
                            'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
                            'TRAVEL_MOTIVE_1', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM', 'sido_gungu_list'])
    for i in list(data['TRAVEL_ID'].unique()):
        temp_df = data[data['TRAVEL_ID'] == i]
        temp_df1 = temp_df[['SIDO', 'GUNGU', 'EUPMYEON']] #각 유저별 방문한 시군구 확인
        temp_df1.reset_index(drop = True, inplace = True)
        sido_gungu_visit = []
        for k in range(len(temp_df1)):
            sido_gungu_visit.append(temp_df1['SIDO'][k] + '+' + temp_df1['GUNGU'][k] + '+' + temp_df1['EUPMYEON'][k])
        sido_gungu_list = list(set(sido_gungu_visit))
        new = temp_df.drop(['SIDO', 'GUNGU', 'EUPMYEON'], axis = 1) #기존 시도, 군구 제외하고
        new = new.head(1)
        new['sido_gungu_list'] = str(sido_gungu_list)
        data1 = pd.concat([data1, new], axis = 0) #새로운 데이터프레임 생성 
    data1.reset_index(drop = True, inplace = True)
    ##########################여행지 정보################################
    # 여행지 정보는 훈련세트에서 가져온다. 방문횟수가 number매개변수 이상인 여행지를 중복없이 담은
    # info 데이터 프레임을 얻는다.
    info = X_new_train[['SIDO', 'VISIT_AREA_NM', 'GUNGU', 'EUPMYEON', 'VISIT_AREA_TYPE_CD','RESIDENCE_TIME_MIN_mean', 'RCMDTN_INTENTION_mean',
    'REVISIT_YN_mean', 'TRAVEL_COMPANIONS_NUM_mean', 'REVISIT_INTENTION_mean']]
    info.drop_duplicates(['VISIT_AREA_NM'], inplace = True)
    ###### n회 이상 관광한 방문지 리스트 생성
    visiting_list = X_new_train[['VISIT_AREA_NM']] #train set에 있는 방문지에 대해서만 2회 이상 방문하였는지 확인
    visiting_list.reset_index(drop = True, inplace = True)
    #데이터 전처리
    dfdf = pd.DataFrame(visiting_list.value_counts(), columns = ['count'])
    dfdf['VISIT_AREA_NM'] = dfdf.index
    dfdf.reset_index(drop = True, inplace = True)
    for i in range(len(dfdf)):
        dfdf['VISIT_AREA_NM'][i] = str(dfdf['VISIT_AREA_NM'][i])
        dfdf['VISIT_AREA_NM'][i] = dfdf['VISIT_AREA_NM'][i].replace("(","").replace(")","").replace(",","").replace("\''","")
        dfdf['VISIT_AREA_NM'][i] = dfdf['VISIT_AREA_NM'][i][1:-1]
    #n회 이상 적용
    dfdf = dfdf[dfdf['count'] >= number] 
    visit_list = list(dfdf['VISIT_AREA_NM']) #visit_list에 n회 이상 방문지 리스트
    #방문지가 n회 이상 방문한 관광지 아니면 제거
    info.reset_index(drop = True, inplace = True)
    for i in range(len(info)):
        if info['VISIT_AREA_NM'][i] not in visit_list:
            info = info.drop([i], axis = 0)
    info.reset_index(drop = True, inplace = True)
    ##########################모델 10개 관광지 추천############################
    # 이 result에 담기는 값은
    result = []
    for i in range(len(data1)):
        #데이터
        # final_df는 predict할 때 사용하는 데이터프레임(가공된 검증세트 같은 느낌)
        # 여기가 좀 이해 안될 수 있는데 훈련세트 안에 있는 관광지 중에서 
        # 이용자가 원하는 장소(시군구)에 있는 관광지가 들어있는 데이터 프레임이 final_df임
        
        # 이걸 위해 간단히 설명을 덧붙이자면 이 모델은 이용자가 시군구를 입력하면 거기에 있는 
        # 여행자가 좋아할만한 다양한 장소들을 추천해주는 모델이 아님 실제로는 훈련세트안에 있는 장소들을
        # 기억해놨다가 이용자가 시군구를 입력하면 해당 시군구에 있는 입력된 관광지들이 쭉 나오고
        # 그 중에서 이용자가 갔을 때 만족도가 높을 것 같은 장소 상위 10개를 추천해주는 모델임
        # 즉 훈련세트를 통해 학습시킨 관광지만 추천해 줄 수 있다.
        
        # 결론적으로 final_df는 이용자가 입력한(여기선 방문한 이지만 입력한으로 본다.) 시군구에 있는
        # 훈련세트를 통해 학습된 관광지들이 전부 담긴 데이터프레임이다.
        
        # 그렇기 때문에 검증세트에서 방문한 시군구가 훈련세트에 입력되어 있지않으면 빈 데이터프레임이된다.
        
        final_df = pd.DataFrame(columns = ['VISIT_AREA_NM', 'SIDO', 'GUNGU', 'EUPMYEON', 'VISIT_AREA_TYPE_CD',
               'TRAVEL_MISSION_PRIORITY', 'GENDER', 'AGE_GRP',
               'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
               'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
               'TRAVEL_MOTIVE_1', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM',
               'RESIDENCE_TIME_MIN_mean', 'RCMDTN_INTENTION_mean', 'REVISIT_YN_mean',
               'TRAVEL_COMPANIONS_NUM_mean', 'REVISIT_INTENTION_mean']) #빈 데이터프레임에 내용 추가
        ####### 시/도 군/구 별 자료 수집
        temp = data1['sido_gungu_list'][i].replace("[","").replace("]","").replace("\'","").replace(", ",",")
        places_list = list(map(str, temp.split(",")))
        for q in places_list:
            sido, gungu, eupmyeon = map(str, q.split("+"))

            info_df = info[(info['SIDO'] == sido) & (info['GUNGU'] == gungu)& (info['EUPMYEON'] == eupmyeon)] 

            # info_df.drop(['SIDO'], inplace = True, axis = 1)
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
           'TRAVEL_MOTIVE_1', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM',
           'RESIDENCE_TIME_MIN_mean', 'RCMDTN_INTENTION_mean', 'REVISIT_YN_mean',
           'TRAVEL_COMPANIONS_NUM_mean', 'REVISIT_INTENTION_mean']] # 변수정렬
            df['VISIT_AREA_TYPE_CD'] = df['VISIT_AREA_TYPE_CD'].astype('string')
            final_df = pd.concat([final_df, df], axis = 0)
        final_df.reset_index(drop = True, inplace = True)
        final_df.drop_duplicates(['VISIT_AREA_NM'], inplace = True)

        #모델 예측
        y_pred = model.predict(final_df)
        y_pred = pd.DataFrame(y_pred, columns = ['y_pred'])
        test_df1 = pd.concat([final_df, y_pred], axis = 1)
        test_df1.sort_values(by = ['y_pred'], axis = 0, ascending=False, inplace = True) # 예측치가 높은 순대로 정렬

        test_df1 = test_df1.iloc[0:10,] #상위 10개 관광지 추천

        visiting_candidates = list(test_df1['VISIT_AREA_NM']) # 모델이 추천한 관광지들을 리스트 형태로 변환

       # 유저정보와 추천 관광지
       # 여기서 다시 유저정보를 나눠주고
        test_df2 = test_df1[['TRAVEL_MISSION_PRIORITY', 'GENDER', 'AGE_GRP',
                            'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
                            'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
                            'TRAVEL_MOTIVE_1', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM']]
        # 위에서 말했듯이 이용자(검증세트)가 입력한(방문한) 시군구가 훈련세트에 없는 경우가 있기 때문에
        # 그런 경우에는 len(test_df2)가 0이 될 수 있다. 이 때는 빈리스트를 result에 담는다.
        # 그 밖에 값이 있으면 test_df2의 열이름을 리스트로 rec에 담고 추천장소(이용자가 입력한
        # 시군구에 있는 관광지(근데 이 관광지는 훈련세트에 있음))를 append해준다. 그 후 result에 추가한다.
        # result는 data1의 길이와 같은 길이를 갖는다. data1은 이용자(검증세트)의 id를 유니크로 뽑아낸 길이
        
        # 결론만 말하면 result는 이용자의 유저정보와 추천 관광지가 리스트형태로 담겨 있는 리스트이다.
        if len(test_df2) == 0:
            rec = []
            result.append(rec)
        else:

            rec = test_df2.iloc[0].to_list()

            rec.append(visiting_candidates)

            result.append(rec)
    # result를 final_df라는 데이터프레임에 담아주고
    final_df = pd.DataFrame(result,
                        columns = ['TRAVEL_MISSION_PRIORITY', 'GENDER', 'AGE_GRP',
                        'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
                        'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
                        'TRAVEL_MOTIVE_1', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM', 'recommend_result_place'])
    # 그 중 추천 관광지만 가져온다
    # travel_id는 data1의 travel_id 즉 중복없는 이용자 아이디가 담긴다.
    final_df = final_df[['recommend_result_place']]
    travel_id = data1[['TRAVEL_ID']]
    travel_id.reset_index(drop = True, inplace = True)
    # 여기서 final_df에는 이용자 travel_id와 각각에게 추천된 추천 관광지가 담긴다.
    final_df = pd.concat([travel_id, final_df], axis = 1)

    #추천지 10개 미만인 여행 ID확인
    travel_id_list = [] # 추천지 10개 미만인 여행id가 담김
    for i in range(len(final_df)):
        recommend_list = final_df['recommend_result_place'][i]
        if str(recommend_list).count(',') < 9:
            travel_id_list.append(final_df['TRAVEL_ID'][i])
        if pd.isna(str(recommend_list)):
            travel_id_list.append(final_df['TRAVEL_ID'][i])
    ## 군구 리스트
    # 검증세트의 군구열이 구, 시, 군으로 끝나면 마지막 글자를 제외한다.
    places = list(set(globals()['Train_'+str(j+1)]['GUNGU']))
    for i in range(len(places)):
        if places[i][-1] == '구' or places[i][-1] == '시' or places[i][-1] == '군':
            places[i] = places[i][:-1]

    ###############################################################
    ######## 최종 성능 평가 #########################################
    #########################################################
    # visit_list에는 훈련세트로 학습한 중복없는 number매개변수 이상 방문된 장소가 담김
    visit_list = list(info['VISIT_AREA_NM'])
    # 검증세트에 만족도(타겟)열 추가
    globals()['Train_'+str(j+1)]['DGSTFN'] = globals()['target_'+str(j+1)]

    for i in list(globals()['Train_'+str(j+1)]['TRAVEL_ID'].unique()):

        #추천한 방문지가 10개 미만이면 0
        if i in travel_id_list:
            recall_10_list.append(0)
            continue
        # 그냥 검증세트 중 추천관광지가 10개 이상인 여행id
        satisfied = globals()['Train_'+str(j+1)][globals()['Train_'+str(j+1)]['TRAVEL_ID'] == i]
        satisfied.reset_index(drop = True, inplace = True) 

        # 그 중 만족도 4이상인 여행id만 따로 빼서 satisfied1에 담음
        satisfied1 = satisfied[satisfied['DGSTFN'] >=4 ] #만족의 기준은 4이상 일때만 만족이라고 정의
        if len(satisfied1) == 0: # 유저가 만족한 관광지가 하나도 없으면 recall@10은 어차피 0
            recall_10_list.append(0)
            continue
        else:
            # 이용자(검증세트)가 실제 방문한 장소 중 만족도(타겟)가 4이상인 장소가 item_list에 담겼다.
            item_list = satisfied1['VISIT_AREA_NM']


        item_list = list(set(item_list))


    #final_df의 추천지 10개랑 비교

        recommend_list = final_df[final_df['TRAVEL_ID'] == i]['recommend_result_place']

        # 검증세트가 실제 방문했을 때 만족도가 4이상인 관광지가 추천지 10개 안에 있으면 summ에 +1한다.
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
        # 풀어 말하면 어떤 사람이 특정 시군구에 갔을 때 실제로 방문해서 만족했던 곳이 추천지 10개 안에
        # 몇 개가 있는지에 대한 비율
        recall10_for_1user = summ / min(10, len(satisfied1)) #recall@10 산식
        if recall10_for_1user > 1:
            recall10_for_1user = 1
        # 얘들을 append해주면 recall_10_list에는 검증세트 모두에 대한 예측비율이 담김
        recall_10_list.append(recall10_for_1user)
    globals()['Train_'+str(j+1)].drop(['DGSTFN'], axis = 1, inplace = True) #globals 함수 오류 해결하기 위한 코드
    
    # 이걸 다시 평균내면 한 폴드의 전체 예측성능이 나온다.
    recall_for_one_cv = sum(recall_10_list) / len(recall_10_list) #한 fold에 대한 recall@10값 추출
    
    # 이걸 다시 cv수 만큼 리스트에 담아주고
    final_recall.append(recall_for_one_cv)

# 평균내 주면 전체 검증세트에 대한 최종 예측성능이 나온다.
recallat10 = sum(final_recall) / len(final_recall)
print('이번 결과는:', recallat10)
###################### hyperparameter 바꾸면 여기도 수정해야 ######################
###################################################################################
print('이번 결과의 parameter은: ', 'n_estimators:', n_estimators)
if recallat10 > initial:
    initial = recallat10
    print('신기록!:', initial)
    print('n_estimators:', n_estimators)
    final_estimator = n_estimators

 #####################################################################################
print('최종 parameter은 :',  final_estimator)
#%%
globals()['Train_'+str(j+1)].drop(['TRAVEL_ID','TRAVELER_ID', 'REVISIT_INTENTION', 'RCMDTN_INTENTION','RESIDENCE_TIME_MIN',
            'REVISIT_YN','INCOME'], axis = 1, inplace = True)
globals()['Train_'+str(j+1)]['VISIT_AREA_TYPE_CD'] = globals()['Train_'+str(j+1)]['VISIT_AREA_TYPE_CD'].astype('string')


y_val = globals()['Train_'+str(j+1)]['DGSTFN']
X_val = globals()['Train_'+str(j+1)].drop(['DGSTFN'], axis = 1)

# 모델의 예측값과 실제 레이블 가져오기
y_pred2 = model.predict(X_val)
y_true = y_val  # 실제 레이블

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true, y_pred2)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred2)

from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred2)

print('평균제곱오차: ', mse)
print('평균절대오차: ', mae)
print('R-squared: ', r2)

