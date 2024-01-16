# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 11:52:56 2024

@author: InQ
"""
import pandas as pd

path ='.'

Train1 = pd.read_csv(path + '/source/관광지 추천시스템 Trainset_final.csv')
Train2 = pd.read_csv(path + '/source/관광지 추천시스템 Trainset_final_ac.csv')
Train3 = pd.read_csv(path + '/source/관광지 추천시스템 Trainset_final_re.csv')

test1 = pd.read_csv(path + '/source/관광지 추천시스템 Testset_final.csv')
test2 = pd.read_csv(path + '/source/관광지 추천시스템 Testset_final_ac.csv')
test3 = pd.read_csv(path + '/source/관광지 추천시스템 Testset_final_re.csv')

info1 = pd.read_csv(path + '/source/관광지 추천시스템 여행지 정보 방문 2회 이상.csv')
info2 = pd.read_csv(path + '/source/관광지 추천시스템 여행지 정보 방문 2회 이상_ac.csv')
info3 = pd.read_csv(path + '/source/관광지 추천시스템 여행지 정보 방문 2회 이상_re.csv')
#%%
Train = pd.concat([Train1, Train2, Train3], axis=0, ignore_index=True)
test = pd.concat([test1, test2, test3], axis=0, ignore_index=True)
info = pd.concat([info1, info2, info3], axis=0, ignore_index=True)
#%%
Train.to_csv(path + '/source/관광지 추천시스템 Trainset_final_total.csv', index = False)
test.to_csv(path + '/source/관광지 추천시스템 Testset_final_total.csv', index = False)
info.to_csv(path + '/source/관광지 추천시스템 여행지 정보 방문 2회 이상_total.csv', index = False)

#%%
info.isna().sum()
