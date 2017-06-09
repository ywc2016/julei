# -*- coding: utf-8 -*-
"""
Created on Fri May 26 14:43:03 2017

@author: lixixi
"""

# 流失度管理
import pandas as pd
import matplotlib.pyplot as plt
# from optimazation_k import *
import math
from numpy import *
from sklearn.cluster import KMeans


# 对数据进行归一化处理
# 传入的是series数据类型
def normalization(date_series):
    data_tolist = date_series.tolist()
    data_norm_list = data_tolist
    min_data = min(data_tolist)
    max_data = max(data_tolist)
    for index, data in enumerate(data_tolist):
        if ((max_data - min_data) > 0):
            norm_data = float((data - min_data)) / float((max_data - min_data))
            data_norm_list[index] = norm_data
        else:
            data_norm_list[index] = 0
    return data_norm_list



loss_data_pd = pd.read_excel(u'C:/Users/Administrator/Desktop/indicators_of_customer_value_and_loss.xlsx', sheetname='Sheet1')
# 提取流失值
extract_data_pd = loss_data_pd[['cons_cycle', 'late_to_today']]
# 对流失度画出分布图
data_pd = extract_data_pd[['cons_cycle', 'late_to_today']]
# 将数据框转化为matrix
# data_matrix=mat(data_pd)
# 做一个聚类分析
# dataMat =mat(loadDataSet(r'D:\book_data\testdata.txt'))
# myCentroids,clustAssing = kMeans(dataMat,11)
# c,b=biKmeans(data_matrix, 20, distMeas=distEclud)

plt.scatter(normalization(data_pd['late_to_today']), normalization(data_pd['cons_cycle']))
plt.xlabel('late_to_today_by_month')
plt.ylabel('cons_cycle')


# 将数据做标准化处理
data_norm_pd = pd.DataFrame(columns=['late_to_today_by_month_normde', 'cons_cycle_normed'])
data_norm_pd['late_to_today_by_month_normde'] = normalization(data_pd['late_to_today'])
data_norm_pd['cons_cycle_normed'] = normalization(data_pd['cons_cycle'])
dat_norm_matrix = mat(data_norm_pd)

# c,b=biKmeans(dat_norm_matrix, 20, distMeas=distEclud)
# 将数据转化为numpy
# 通过训练，我们的k=3的最为合适
data_arry = array(data_norm_pd)
# 聚类的个数
num_clusters = 2
km = KMeans(n_clusters=num_clusters)

km.fit(data_arry)
clusters_lis = km.labels_.tolist()
# 将类标签加入到我们的原始数据框中，然后分类汇总‘
data_pd['label'] = clusters_lis
# 根据类标号分类汇总
mean_lable = data_pd.groupby(['label']).mean()
plt.scatter(mean_lable['late_to_today'], mean_lable['cons_cycle'])
plt.xlabel('late_to_today_by_month')
plt.ylabel('cons_cycle')
# for label,multi_group data_pd.groupby(['label']):

