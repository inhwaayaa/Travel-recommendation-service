
import streamlit as st
import pandas as pd
import joblib
import warnings
import folium
from streamlit_folium import folium_static
warnings.filterwarnings('ignore') #warning 문구 제거
pd.set_option('display.max_columns', None)

#%%
path ='.'
modeld = joblib.load(path + '/ML/catboost_model_D_travel_total.pkl')
place_info = pd.read_csv(path + '/source/관광지 추천시스템 여행지 정보 방문 2회 이상_total.csv')
#%%
st.set_page_config(
    layout="wide", page_title="여행지 추천", page_icon="./images/YS_logo.png"
)

#%%
# CSS 정의
custom_css = """
<style>
    .custom-sidebar {
        background-color: #333;
        padding: 20px;
        color: #fff;
    }
    .custom-form {
        margin-bottom: 20px;
    }
    
    a {
       text-decoration: None;
       color: #000 !important;
    }
    iframe {
        width: 100%
    }
</style>
"""

# 사용자 정의 CSS 적용
st.markdown(custom_css, unsafe_allow_html=True)

# 여행지 추천 서비스 Streamlit 코드
st.title('여행지 추천 서비스')

#%%
# 페이지 상단 메뉴
st.markdown('''##### <span style="color:gray">Customized travel recommendation service using machine learning</span>
            ''', unsafe_allow_html=True)
                
travel, accom, rest = st.tabs(["여행지", "숙소", "음식점"])

a,b,c = (travel, accom, rest)
#%%

# 사용자 입력 폼
st.sidebar.markdown("<div class='custom-sidebar'>여행지 추천 서비스</div>", unsafe_allow_html=True)
#st.sidebar.header('사용자 정보 입력')
st.sidebar.markdown("---")
tmis_list1 = ['쇼핑','테마파크, 놀이시설, 동/식물원 방문','역사 유적지 방문','시티투어',
              '야외 스포츠, 레포츠 활동','지역 문화예술/공연/전시시설 관람','유흥/오락(나이트라이프)',
              '캠핑','지역 축제/이벤트 참가','온천/스파','교육/체험 프로그램 참가',
               '드라마 촬영지 방문','종교/성지 순례']
tmis_list2 = ['Well-ness 여행','SNS 인생샷 여행',
              '호캉스 여행','신규 여행지 발굴','반려동물 동반 여행','인플루언서 따라하기 여행',
              '친환경 여행(플로깅 여행)','등반 여행']
tmis_list = tmis_list1+tmis_list2


tmis_input = st.sidebar.selectbox('여행 테마', tmis_list)

for i,v in enumerate(tmis_list1, start=1):
    if tmis_input == v:
        tmis = i 
for i,v in enumerate(tmis_list2, start=21):
    if tmis_input == v:
        tmis = i 

gender = st.sidebar.selectbox('성별', ['남성', '여성'])

age = st.sidebar.number_input("나이", min_value=1, max_value=100, value=20, step=1)

if age<10:
    age_grp = 0
elif age<20:
    age_grp = 10
elif age<30:
    age_grp = 20
elif age<40:
    age_grp = 30
elif age<50:
    age_grp = 40
elif age<60:
    age_grp = 50
else:
    age_grp = 60
# 데이터 불러오기
location_data = place_info

# 사용자로부터 시/도 선택
sido_data = location_data['SIDO'].unique()
sorted_sido = pd.Series(sido_data).sort_values()
sido = st.sidebar.selectbox('시/도를 선택하세요', sorted_sido)

# 선택한 시/도에 속하는 군/구 목록 가져오기

gungu_data = location_data[location_data['SIDO'] == sido]['GUNGU'].unique()
sorted_gungu = pd.Series(gungu_data).sort_values()
gungu = st.sidebar.selectbox('군/구를 선택하세요', sorted_gungu)

#
eupmyeon_data = location_data[(location_data['SIDO'] == sido) & (location_data['GUNGU'] == gungu)]['EUPMYEON'].unique()
sorted_eupmyeon = pd.Series(eupmyeon_data).sort_values()
tot_eup = pd.Series(['전체'])
sorted_eupmyeon = pd.concat([tot_eup, sorted_eupmyeon], axis=0)
eupmyeon = st.sidebar.selectbox('읍/면을 선택하세요', sorted_eupmyeon, )

st.sidebar.markdown("---")

tsy_1 = st.sidebar.slider('자연 vs 도시', 1, 7, 4)
tsy_2 = st.sidebar.slider('숙박 vs 당일', 1, 7, 4)
# tsy_3 = st.sidebar.slider('새로운 지역 vs 익숙한 지역', 1, 7, 4)
# tsy_4 = st.sidebar.slider('편하지만 비싼 숙소 vs 불편하지만 저렴한 숙소', 1, 7, 4)
tsy_5 = st.sidebar.slider('휴양/휴식 vs 체험활동', 1, 7, 4)
# tsy_6 = st.sidebar.slider('잘 알려지지 않은 방문지 vs 알려진 방문지', 1, 7, 4)
# tsy_7 = st.sidebar.slider('계획 vs 즉흥', 1, 7, 4)
# tsy_8 = st.sidebar.slider('사진촬영 안중요 vs 중요', 1, 7, 4)
tsy_3 = 4
tsy_4 = 4
tsy_6 = 4
tsy_7 = 4
tsy_8 = 4
st.sidebar.markdown("---")
tmt_list = ['일상적인 환경 및 역할에서의 탈출, 지루함 탈피', '쉴 수 있는 기회, 육체 피로 해결 및 정신적인 휴식',
                             '여행 동반자와의 친밀감 및 유대감 증진', '진정한 자아 찾기 또는 자신을 되돌아볼 기회 찾기',
                             'SNS 사진 등록 등', '운동, 건강 증진 및 충전', '새로운 경험 추구', '역사 탐방, 문화적 경험 등 교육적 동기',
                             '특별한 목적(칠순여행, 신혼여행, 수학여행, 인센티브여행)', '기타']
tmt_input = st.sidebar.selectbox('여행 동기', tmt_list)
for i,v in enumerate(tmt_list, start=1):
    if tmt_input == v:
        tmt = i

tnm = st.sidebar.number_input("여행 횟수", min_value=1, max_value=100, value=1, step=1)
tcpn = st.sidebar.number_input("동반자 수", min_value=1, max_value=20, value=1, step=1)
st.sidebar.markdown("---")
#유저정보
data = pd.DataFrame({
    'TRAVEL_MISSION_PRIORITY': [tmis],
    'GENDER': [gender],
    'AGE_GRP': [age_grp],
    'SIDO' : [sido],
    'GUNGU': [gungu],
    'EUPMYEON': [eupmyeon],
    # 'INCOME': [income],
    'TRAVEL_STYL_1' : [tsy_1],
    'TRAVEL_STYL_2' : [tsy_2],
    'TRAVEL_STYL_3' : [tsy_3],
    'TRAVEL_STYL_4' : [tsy_4],
    'TRAVEL_STYL_5': [tsy_5],
    'TRAVEL_STYL_6': [tsy_6],
    'TRAVEL_STYL_7': [tsy_7],
    'TRAVEL_STYL_8': [tsy_8],
    'TRAVEL_MOTIVE_1': [tmt],
    'TRAVEL_NUM': [tnm],
    'TRAVEL_COMPANIONS_NUM': [tcpn]
    # ... 추가적인 사용자 정보
})
#%%
#여행지 정보
if st.sidebar.button("여행지 추천"):
    if travel:
        info = pd.read_csv(path + '/source/관광지 추천시스템 여행지 정보 방문 2회 이상.csv')
        
        
        
        final_df = pd.DataFrame(columns = ['VISIT_AREA_NM', 'SIDO', 'GUNGU', 'EUPMYEON', 'VISIT_AREA_TYPE_CD',
               'TRAVEL_MISSION_PRIORITY', 'GENDER', 'AGE_GRP',
               'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
               'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
               'TRAVEL_MOTIVE_1', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM', 'RESIDENCE_TIME_MIN_mean', 'RCMDTN_INTENTION_mean',
               'REVISIT_YN_mean', 'TRAVEL_COMPANIONS_NUM_mean', 'REVISIT_INTENTION_mean']) #빈 데이터프레임에 내용 추가
        ####### 시/도 군/구 별 자료 수집
        
        if eupmyeon == '전체':
            info_df = info[(info['SIDO'] == sido) & (info['GUNGU'] == gungu)] 
        else:
            info_df = info[(info['SIDO'] == sido) & (info['GUNGU'] == gungu) & (info['EUPMYEON'] == eupmyeon)] 
        
        info_df.reset_index(inplace = True, drop = True)
        data2 = data.drop(['SIDO','GUNGU', 'EUPMYEON'], axis =1)
        user_df = pd.DataFrame([data2.iloc[0].to_list()]*len(info_df), columns = ['TRAVEL_MISSION_PRIORITY', 'GENDER', 'AGE_GRP',
                             'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
                             'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
                             'TRAVEL_MOTIVE_1', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM'])
        df = pd.concat([user_df, info_df], axis = 1)
        df = df[['VISIT_AREA_NM', 'SIDO', 'GUNGU', 'EUPMYEON', 'VISIT_AREA_TYPE_CD',
        'TRAVEL_MISSION_PRIORITY', 'GENDER', 'AGE_GRP',
        'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
        'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
        'TRAVEL_MOTIVE_1', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM', 'RESIDENCE_TIME_MIN_mean', 'RCMDTN_INTENTION_mean',
        'REVISIT_YN_mean', 'TRAVEL_COMPANIONS_NUM_mean', 'REVISIT_INTENTION_mean']] # 변수정렬
        df['VISIT_AREA_TYPE_CD'] = df['VISIT_AREA_TYPE_CD'].astype('string')
        final_df = pd.concat([final_df, df], axis = 0)
        final_df.reset_index(drop = True, inplace = True)
        final_df.drop_duplicates(['VISIT_AREA_NM'], inplace = True)
        
        #모델 예측
        
        y_pred = modeld.predict(final_df)
        y_pred = pd.DataFrame(y_pred, columns = ['y_pred'])
        test_df1 = pd.concat([final_df, y_pred], axis = 1)    
        test_df1.sort_values(by = ['y_pred'], axis = 0, ascending=False, inplace = True) # 예측치가 높은 순대로 정렬
        
        # 추천 여행지
        recomand10 = test_df1[['VISIT_AREA_NM','y_pred']].head(10)
        
        placelist = list(recomand10['VISIT_AREA_NM'])
        satlist = list(recomand10['y_pred'])
        linklist = []
        for i in range(len(placelist)):
            if len(placelist[i].split(' ')) >=2:
                placelist2='%20'.join(placelist[i].split(' '))        
                link = f"[{placelist2}](https://map.naver.com/p/search/{placelist2})"
                
            else:        
                link = f"[{placelist[i]}](https://map.naver.com/p/search/{placelist[i]})"
            linklist.append(link)
        recomand10_A = pd.DataFrame({
            '추천 여행지': placelist,
            '예상 만족도': satlist    
        })
        with travel:
            con1,con2,con3 = st.columns([0.4,0.2,0.2])
            
            with con1:
                coords = pd.read_csv(path + '/source/tn_visit_area_info_방문지정보_total.csv')
                coords.dropna(subset = ['LOTNO_ADDR'], inplace = True)
                coords.reset_index(drop = True, inplace=True)
            
                unique = coords[['VISIT_AREA_NM','X_COORD','Y_COORD']].drop_duplicates(subset=['VISIT_AREA_NM'])
                
            
                merged_df = pd.merge(final_df, unique[['VISIT_AREA_NM', 'X_COORD', 'Y_COORD']], on='VISIT_AREA_NM', how='left')
            
                # NaN 값이 있는 행 삭제
                merged_df = merged_df.dropna(subset=['X_COORD', 'Y_COORD'])
                
                # 'VISIT_AREA_NM'을 기준으로 recomand10과 merged_df를 조인
                selected_places = pd.merge(recomand10, merged_df[['VISIT_AREA_NM', 'X_COORD', 'Y_COORD']], on='VISIT_AREA_NM', how='left')
                
                # 'X_COORD', 'Y_COORD' 컬럼이 없는 행은 제거
                selected_places = selected_places.dropna(subset=['X_COORD', 'Y_COORD'])
                
                # 지도 초기화
                st.subheader("추천 여행지 지도")
                selected_places['X_COORD'] = pd.to_numeric(selected_places['X_COORD'], errors='coerce')
                selected_places['Y_COORD'] = pd.to_numeric(selected_places['Y_COORD'], errors='coerce')
                
                # Folium 지도 코드 추가
                # m = folium.Map(location=[filtered_data['Latitude'].mean(), filtered_data['Longitude'].mean()], zoom_start=15)
                
                # 변환 후 평균 계산
                m = folium.Map(location=[selected_places['Y_COORD'].mean(), selected_places['X_COORD'].mean()], zoom_start=11)
                
                # 추천 여행지를 지도에 마커로 추가
                for index, row in selected_places.iterrows():
                    folium.Marker([row['Y_COORD'], row['X_COORD']], popup=row['VISIT_AREA_NM']).add_to(m)
                
                # 지도를 Streamlit에 표시
                folium_static(m)        
                
            with con2:
                st.subheader("여행지 추천 결과")
                if not recomand10.empty:
                    #st.write(recomand10_A)
                    st.dataframe(recomand10_A, width=400)
                    #st.write(recomand10_link)
                    #st.write(recomand10_link[['추천 여행지', '예상 만족도', '지도 보기']])
                    
                else:
                    st.warning("조건에 맞는 여행지가 없습니다.")
            with con3:
                st.subheader("여행지 정보")
                for i in range(len(linklist)):
                    if '%20' in linklist[i]:
                        count = linklist[i].count('%20')                
                        if count == 2:
                            for j in range(count-1):
                                index = linklist[i].find('%20')
                                linklist[i] = linklist[i][:index] + ' ' + linklist[i][index+3:]
                        elif count == 4:
                            for j in range(count-2):
                                index = linklist[i].find('%20')
                                linklist[i] = linklist[i][:index] + ' ' + linklist[i][index+3:]
                        elif count == 6:
                            for j in range(count-3):
                                index = linklist[i].find('%20')
                                linklist[i] = linklist[i][:index] + ' ' + linklist[i][index+3:]
                        
                    st.write(linklist[i])
                    
                
        # 장소 띄어쓰기 확인하는 코드        
        # for i in range(len(info)):
        #     count2 = info['VISIT_AREA_NM'][i].count(' ')
        #     if count2 >=3:
        #         print('띄어쓰기 3개 이상', info['VISIT_AREA_NM'][i])
    
    if accom:   
        info = pd.read_csv(path + '/source/관광지 추천시스템 여행지 정보 방문 2회 이상_ac.csv')
        
        
        
        final_df = pd.DataFrame(columns = ['VISIT_AREA_NM', 'SIDO', 'GUNGU', 'EUPMYEON', 'VISIT_AREA_TYPE_CD',
               'TRAVEL_MISSION_PRIORITY', 'GENDER', 'AGE_GRP',
               'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
               'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
               'TRAVEL_MOTIVE_1', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM', 'RESIDENCE_TIME_MIN_mean', 'RCMDTN_INTENTION_mean',
               'REVISIT_YN_mean', 'TRAVEL_COMPANIONS_NUM_mean', 'REVISIT_INTENTION_mean']) #빈 데이터프레임에 내용 추가
        ####### 시/도 군/구 별 자료 수집
        
        if eupmyeon == '전체':
            info_df = info[(info['SIDO'] == sido) & (info['GUNGU'] == gungu)] 
        else:
            info_df = info[(info['SIDO'] == sido) & (info['GUNGU'] == gungu) & (info['EUPMYEON'] == eupmyeon)] 
        
        info_df.reset_index(inplace = True, drop = True)
        data2 = data.drop(['SIDO','GUNGU', 'EUPMYEON'], axis =1)
        user_df = pd.DataFrame([data2.iloc[0].to_list()]*len(info_df), columns = ['TRAVEL_MISSION_PRIORITY', 'GENDER', 'AGE_GRP',
                             'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
                             'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
                             'TRAVEL_MOTIVE_1', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM'])
        df = pd.concat([user_df, info_df], axis = 1)
        df = df[['VISIT_AREA_NM', 'SIDO', 'GUNGU', 'EUPMYEON', 'VISIT_AREA_TYPE_CD',
        'TRAVEL_MISSION_PRIORITY', 'GENDER', 'AGE_GRP',
        'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
        'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
        'TRAVEL_MOTIVE_1', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM', 'RESIDENCE_TIME_MIN_mean', 'RCMDTN_INTENTION_mean',
        'REVISIT_YN_mean', 'TRAVEL_COMPANIONS_NUM_mean', 'REVISIT_INTENTION_mean']] # 변수정렬
        df['VISIT_AREA_TYPE_CD'] = df['VISIT_AREA_TYPE_CD'].astype('string')
        final_df = pd.concat([final_df, df], axis = 0)
        final_df.reset_index(drop = True, inplace = True)
        final_df.drop_duplicates(['VISIT_AREA_NM'], inplace = True)
        
        #모델 예측
        
        y_pred = modeld.predict(final_df)
        y_pred = pd.DataFrame(y_pred, columns = ['y_pred'])
        test_df1 = pd.concat([final_df, y_pred], axis = 1)    
        test_df1.sort_values(by = ['y_pred'], axis = 0, ascending=False, inplace = True) # 예측치가 높은 순대로 정렬
        
        # 추천 여행지
        recomand10 = test_df1[['VISIT_AREA_NM','y_pred']].head(10)
        
        placelist = list(recomand10['VISIT_AREA_NM'])
        satlist = list(recomand10['y_pred'])
        linklist = []
        for i in range(len(placelist)):
            if len(placelist[i].split(' ')) >=2:
                placelist2='%20'.join(placelist[i].split(' '))        
                link = f"[{placelist2}](https://map.naver.com/p/search/{placelist2})"
                
            else:        
                link = f"[{placelist[i]}](https://map.naver.com/p/search/{placelist[i]})"
            linklist.append(link)
        recomand10_A = pd.DataFrame({
            '추천 여행지': placelist,
            '예상 만족도': satlist    
        })
        with accom:
            con1,con2,con3 = st.columns([0.4,0.2,0.2])
            
            with con1:
                coords = pd.read_csv(path + '/source/tn_visit_area_info_방문지정보_total.csv')
                coords.dropna(subset = ['LOTNO_ADDR'], inplace = True)
                # 수작업..            
                coords.reset_index(drop = True, inplace=True)
                
                unique = coords[['VISIT_AREA_NM','X_COORD','Y_COORD']].drop_duplicates(subset=['VISIT_AREA_NM'])
                
            
                merged_df = pd.merge(final_df, unique[['VISIT_AREA_NM', 'X_COORD', 'Y_COORD']], on='VISIT_AREA_NM', how='left')
            
                # NaN 값이 있는 행 삭제
                merged_df = merged_df.dropna(subset=['X_COORD', 'Y_COORD'])
                
                # 'VISIT_AREA_NM'을 기준으로 recomand10과 merged_df를 조인
                selected_places = pd.merge(recomand10, merged_df[['VISIT_AREA_NM', 'X_COORD', 'Y_COORD']], on='VISIT_AREA_NM', how='left')
                
                # 'X_COORD', 'Y_COORD' 컬럼이 없는 행은 제거
                selected_places = selected_places.dropna(subset=['X_COORD', 'Y_COORD'])
                
                # 지도 초기화
                st.subheader("추천 여행지 지도")
                selected_places['X_COORD'] = pd.to_numeric(selected_places['X_COORD'], errors='coerce')
                selected_places['Y_COORD'] = pd.to_numeric(selected_places['Y_COORD'], errors='coerce')
                
                # Folium 지도 코드 추가
                # m = folium.Map(location=[filtered_data['Latitude'].mean(), filtered_data['Longitude'].mean()], zoom_start=15)
                
                # 변환 후 평균 계산
                m = folium.Map(location=[selected_places['Y_COORD'].mean(), selected_places['X_COORD'].mean()], zoom_start=11)
                
                # 추천 여행지를 지도에 마커로 추가
                for index, row in selected_places.iterrows():
                    folium.Marker([row['Y_COORD'], row['X_COORD']], popup=row['VISIT_AREA_NM']).add_to(m)
                
                # 지도를 Streamlit에 표시
                folium_static(m)        
                
            with con2:
                st.subheader("숙소 추천 결과")
                if not recomand10.empty:
                    st.dataframe(recomand10_A, width=400)
                    #st.write(recomand10_link)
                    #st.write(recomand10_link[['추천 여행지', '예상 만족도', '지도 보기']])
                    
                else:
                    st.warning("조건에 맞는 여행지가 없습니다.")
            with con3:
                st.subheader("숙소 정보")
                for i in range(len(linklist)):
                    if '%20' in linklist[i]:
                        count = linklist[i].count('%20')                
                        if count == 2:
                            for j in range(count-1):
                                index = linklist[i].find('%20')
                                linklist[i] = linklist[i][:index] + ' ' + linklist[i][index+3:]
                        elif count == 4:
                            for j in range(count-2):
                                index = linklist[i].find('%20')
                                linklist[i] = linklist[i][:index] + ' ' + linklist[i][index+3:]
                        elif count == 6:
                            for j in range(count-3):
                                index = linklist[i].find('%20')
                                linklist[i] = linklist[i][:index] + ' ' + linklist[i][index+3:]                    
                    st.write(linklist[i])
                    
    if rest:   
        info = pd.read_csv(path + '/source/관광지 추천시스템 여행지 정보 방문 2회 이상_re.csv')
        
        
        final_df = pd.DataFrame(columns = ['VISIT_AREA_NM', 'SIDO', 'GUNGU', 'EUPMYEON', 'VISIT_AREA_TYPE_CD',
               'TRAVEL_MISSION_PRIORITY', 'GENDER', 'AGE_GRP',
               'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
               'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
               'TRAVEL_MOTIVE_1', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM', 'RESIDENCE_TIME_MIN_mean', 'RCMDTN_INTENTION_mean',
               'REVISIT_YN_mean', 'TRAVEL_COMPANIONS_NUM_mean', 'REVISIT_INTENTION_mean']) #빈 데이터프레임에 내용 추가
        ####### 시/도 군/구 별 자료 수집
        
        if eupmyeon == '전체':
            info_df = info[(info['SIDO'] == sido) & (info['GUNGU'] == gungu)] 
        else:
            info_df = info[(info['SIDO'] == sido) & (info['GUNGU'] == gungu) & (info['EUPMYEON'] == eupmyeon)] 
        
        info_df.reset_index(inplace = True, drop = True)
        data2 = data.drop(['SIDO','GUNGU', 'EUPMYEON'], axis =1)
        user_df = pd.DataFrame([data2.iloc[0].to_list()]*len(info_df), columns = ['TRAVEL_MISSION_PRIORITY', 'GENDER', 'AGE_GRP',
                             'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
                             'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
                             'TRAVEL_MOTIVE_1', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM'])
        df = pd.concat([user_df, info_df], axis = 1)
        df = df[['VISIT_AREA_NM', 'SIDO', 'GUNGU', 'EUPMYEON', 'VISIT_AREA_TYPE_CD',
        'TRAVEL_MISSION_PRIORITY', 'GENDER', 'AGE_GRP',
        'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
        'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
        'TRAVEL_MOTIVE_1', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM', 'RESIDENCE_TIME_MIN_mean', 'RCMDTN_INTENTION_mean',
        'REVISIT_YN_mean', 'TRAVEL_COMPANIONS_NUM_mean', 'REVISIT_INTENTION_mean']] # 변수정렬
        df['VISIT_AREA_TYPE_CD'] = df['VISIT_AREA_TYPE_CD'].astype('string')
        final_df = pd.concat([final_df, df], axis = 0)
        final_df.reset_index(drop = True, inplace = True)
        final_df.drop_duplicates(['VISIT_AREA_NM'], inplace = True)
        
        #모델 예측
        
        y_pred = modeld.predict(final_df)
        y_pred = pd.DataFrame(y_pred, columns = ['y_pred'])
        test_df1 = pd.concat([final_df, y_pred], axis = 1)    
        test_df1.sort_values(by = ['y_pred'], axis = 0, ascending=False, inplace = True) # 예측치가 높은 순대로 정렬
        
        # 추천 여행지
        recomand10 = test_df1[['VISIT_AREA_NM','y_pred']].head(10)
        
        placelist = list(recomand10['VISIT_AREA_NM'])
        satlist = list(recomand10['y_pred'])
        linklist = []
        for i in range(len(placelist)):
            if len(placelist[i].split(' ')) >=2:
                placelist2='%20'.join(placelist[i].split(' '))        
                link = f"[{placelist2}](https://map.naver.com/p/search/{placelist2})"
                
            else:        
                link = f"[{placelist[i]}](https://map.naver.com/p/search/{placelist[i]})"
            linklist.append(link)
        recomand10_A = pd.DataFrame({
            '추천 여행지': placelist,
            '예상 만족도': satlist    
        })            
        with rest:
            con1,con2,con3 = st.columns([0.4,0.2,0.2])
            
            with con1:
                coords = pd.read_csv(path + '/source/tn_visit_area_info_방문지정보_total2.csv')                
                
                coords.dropna(subset = ['LOTNO_ADDR'], inplace = True)
                coords.reset_index(drop = True, inplace=True)
            
                unique = coords[['VISIT_AREA_NM','X_COORD','Y_COORD']].drop_duplicates(subset=['VISIT_AREA_NM'])
                
            
                merged_df = pd.merge(final_df, unique[['VISIT_AREA_NM', 'X_COORD', 'Y_COORD']], on='VISIT_AREA_NM', how='left')
            
                # NaN 값이 있는 행 삭제
                merged_df = merged_df.dropna(subset=['X_COORD', 'Y_COORD'])
                
                # 'VISIT_AREA_NM'을 기준으로 recomand10과 merged_df를 조인
                selected_places = pd.merge(recomand10, merged_df[['VISIT_AREA_NM', 'X_COORD', 'Y_COORD']], on='VISIT_AREA_NM', how='left')
                
                # 'X_COORD', 'Y_COORD' 컬럼이 없는 행은 제거
                selected_places = selected_places.dropna(subset=['X_COORD', 'Y_COORD'])
                
                # 지도 초기화
                st.subheader("추천 여행지 지도")
                selected_places['X_COORD'] = pd.to_numeric(selected_places['X_COORD'], errors='coerce')
                selected_places['Y_COORD'] = pd.to_numeric(selected_places['Y_COORD'], errors='coerce')
                
                # Folium 지도 코드 추가
                # m = folium.Map(location=[filtered_data['Latitude'].mean(), filtered_data['Longitude'].mean()], zoom_start=15)
                
                # 변환 후 평균 계산
                m = folium.Map(location=[selected_places['Y_COORD'].mean(), selected_places['X_COORD'].mean()], zoom_start=11)
                
                # 추천 여행지를 지도에 마커로 추가
                for index, row in selected_places.iterrows():
                    folium.Marker([row['Y_COORD'], row['X_COORD']], popup=row['VISIT_AREA_NM']).add_to(m)
                
                # 지도를 Streamlit에 표시
                folium_static(m)        
                
            with con2:
                st.subheader("식당/카페 추천 결과")
                if not recomand10.empty:
                    st.dataframe(recomand10_A, width=400)
                    #st.write(recomand10_link)
                    #st.write(recomand10_link[['추천 여행지', '예상 만족도', '지도 보기']])
                    
                else:
                    st.warning("조건에 맞는 여행지가 없습니다.")
            with con3:
                st.subheader("식당/카페 정보")
                for i in range(len(linklist)):
                    if '%20' in linklist[i]:
                        count = linklist[i].count('%20')                
                        if count == 2:
                            for j in range(count-1):
                                index = linklist[i].find('%20')
                                linklist[i] = linklist[i][:index] + ' ' + linklist[i][index+3:]
                        elif count == 4:
                            for j in range(count-2):
                                index = linklist[i].find('%20')
                                linklist[i] = linklist[i][:index] + ' ' + linklist[i][index+3:]
                        elif count == 6:
                            for j in range(count-3):
                                index = linklist[i].find('%20')
                                linklist[i] = linklist[i][:index] + ' ' + linklist[i][index+3:]                    
                    st.write(linklist[i])
