# -*- coding: utf-8 -*-
"""
Created on Mon, 2018 April 9th 11:00

@author: wangbl
Purpose: anaylise the network using new methods

"""

import json
import networkx as nx
import pickle
import pandas as pd
import numpy
import os
import matplotlib.pyplot as plt

kinds = 'newdata'
# authorName = 'Enhong Chen'
# authorName = 'Jiawei Han'
# authorName = 'Jian Pei'
# authorName = 'Hui Xiong'
authorName='Geoffrey E. Hinton'
# authorName='Yoshua Bengio'
# authorName='Ilya Sutskever'
# authorName='Michael I. Jordan'
###################20180709write data into csv###################
g = nx.read_gml('trees/newdata/'+str(authorName)+".gml")

triads = {}
for nodes in g.nodes():
    triads.update({nodes:list(nx.triadic_census(g.subgraph([n for n in (g.successors(nodes) and g.predecessors(nodes))])).values())})


data_old = pd.read_csv('log/'+str(kinds)+'/'+str(authorName)+".csv")
# data_old['Unnamed: 0']=data_old['Unnamed: 0'].astype(str)

data = pd.DataFrame.from_dict(triads,orient='index')
data.index.names = ['#author']

data = (pd.concat([data_old,data],axis=1,keys=['#author']))
# print(data)
# data['x1']=data[3]+data[4]+data[5]+data[6]+data[7]+data[10]
# data['x2']=data[9]+data[8]+data[9]+data[11]+data[12]+data[13]+data[14]+data[15]

# cols = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
# data.drop(data.columns[cols],axis=1,inplace=True)


# data.to_csv('log/triads/'+str(authorName)+'_undirected.csv')
######################################

#####################################################
# # print(data)
# def draw_corr(kinds,authorName):
#     '''
#
#     :param df:
#     :param kinds: 不同的建树方法，可选'term5' or 'with_time_limit'
#     :return: 皮尔逊相关系数热力图
#     '''
#     # columns = ['papers', 'pagerank', 'isSameAff', 'isAncestor']
#
#     correlations = data.corr()
#     # print(correlations)#计算变量之间的相关系数矩阵
#     # plot correlation matrix
#     fig = plt.figure() #调用figure创建一个绘图对象
#     ax = fig.add_subplot(111)
#     cax = ax.matshow(correlations, vmin=-1, vmax=1)  #绘制热力图，从-1到1
#     fig.colorbar(cax)  #将matshow生成热力图设置为颜色渐变条
#     ticks = numpy.arange(0,4,1) #生成0-9，步长为1
#     ax.set_xticks(ticks)  #生成刻度
#     ax.set_yticks(ticks)
#     names = ['papers','pagerank',  'isSameAff',  'isAncestor']
#     ax.set_xticklabels(names) #生成x轴标签
#     ax.set_yticklabels(names)
#     plt.savefig('log/triads/'+str(authorName)+'corr.png')#默认为term5的节点，不考虑时间因素的情况
#     plt.show()
# # draw_corr(kinds,authorName)

###############5/4/2018#######################################zhushi####################
# data = pd.read_csv('log/triads/'+str(authorName)+'_undirected.csv')

# data = pd.read_csv('triads(undir)/piority_only1and0/'+str(authorName)+'twoFeatures_undirected.csv')
from collections import defaultdict

from sklearn import svm
from sklearn import linear_model
from sklearn import neural_network
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import mean_squared_error
#############################
from matplotlib import pyplot
import math
import csv
# x1 = np.array(data.iloc[:,2:5])
#
# x2 = np.array(data.iloc[:,8:])
# x = np.hstack((x1,x2))


############20180709 read data, rather than generate data
# headers = ['#author','papers','pagerank','affsChangeTimes','affsChangeFreq','citations','isSameAff','isAncestor','betweenness']
# data= pd.read_csv(open('log/triads/'+str(authorName)+'.csv','r'),index_col=0)
# print(data)

y_papers = np.array(data.iloc[:,1])
print(y_papers)
y_hindex = np.array(data.iloc[:,3])
y_citation = np.array(data.iloc[:,6])


#######################
# data['x1']=pd.Series(np.zeros((len(y),),dtype=int))
#
# data['x2']=pd.Series(np.zeros((len(y),),dtype=int))
# data['x3']=pd.Series(np.zeros((len(y),),dtype=int))
# k=0
# # for i in range(k,8+k):
# #     data['x1'] += data[str(i)]
# # for i in range(9+k,25+k):
# #     data['x2'] += data[str(i)]
# # for i in range(26+k,61+k):
# #     data['x3'] += data[str(i)]
# ###########################
#
# # data['x1']=data[3]+data[4]+data[5]+data[6]+data[7]+data[10]
# # data['x2']=data[9]+data[8]+data[9]+data[11]+data[12]+data[13]+data[14]+data[15]
# # data['x3']=data[]
# cols = []
# # for i in range(5,66):
# #     cols.append(i)
# cols.append(0)
# cols.append(1)
#
# # cols = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
# # data.drop(data.columns[cols],axis=1,inplace=True)
#
#
# # data.to_csv('triads(undir)/piority_only1and0/'+str(authorName)+'_3features_undirected.csv')
# # x= data
# data = data.fillna(0)

##############################
def label_y(y):
    degree=[]
    for i in y:
        if i <= 1:
            degree.append(0)
        elif i < 4:
            degree.append(1)
        elif i < 8:
            degree.append(2)



        # elif i<50:
        #     degree.append(3)

        else:
            degree.append(3)

    return degree




# print(y.shape)
x = np.array(data.iloc[:,2:])
y_classified = np.array(label_y(y_papers))
# print(y_papers)
# print(y_classified.shape)

x_train,x_test,y_train,y_test = train_test_split(x,y_classified,test_size=0.3,random_state=0)



from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report
#
# clf = svm.SVC(kernel='sigmoid').fit(x_train,y_train)#clf1
# clf = DecisionTreeClassifier(random_state=0).fit(x_train,y_train)#clf2
clf = BernoulliNB().fit(x_train,y_train)#clf3
y_pred=clf.predict(x_test)
# scores = cross_val_score(clf,x,y_classified,cv=5,scoring='accuracy')
scores = cross_val_score(clf,x,y_classified,cv=5)
# print(clf.feature_importance_)
# print(scores.mean())
print('BernoulliNB classifier:')
target_names = ['(0,1]','[2,3]','[4,7]','[8+]']
print(classification_report(y_test, y_pred,target_names=target_names))
# 计算准确率
cnt1 = 0
cnt2 = 0
for i in range(len(y_test)):
    if y_pred[i] == y_test[i]:
        cnt1 += 1

    else:
        cnt2 += 1
print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))
############5/4/2018zhushi#################################



################################################
# # clf = linear_model.LinearRegression()
#
# clf = svm.SVR(kernel='poly').fit(x_train,y_train)
# scores = cross_val_score(clf,x,y,cv=5)
# print(scores)
# #
# #
# reg = svm.SVR(kernel='poly').fit(x_train,y_train)
# scores = cross_val_score(clf,x,y,cv=5)
# print(scores)
#
# reg = linear_model.BayesianRidge().fit(x_train,y_train)
# scores = cross_val_score(reg,x,y,cv=5)
# print(scores)
# #
# reg = linear_model.ARDRegression().fit(x_train,y_train)
# scores = cross_val_score(reg,x,y,cv=5)
# print(scores)
#
# reg = neural_network.MLPRegressor(solver='sgd',).fit(x_train,y_train)
# # scores = cross_val_score(reg,x,y,cv=5)
# # print(scores)
# # reg = RandomForestRegressor(max_depth=10,random_state=0).fit(x_train,y_train)
#
# y_pred=reg.predict(x_test)
# mse = mean_squared_error(y_test, y_pred)
# # print(reg.feature_importances_)
# print("MSE: %.4f" % mse)
#
# print ("RMSE:",np.sqrt(mean_squared_error(y_test, y_pred)))
#
# plt.figure()
# plt.plot(range(len(y_pred)),y_pred,'b',label = "predicted")
# plt.plot(range(len(y_pred)),y_test,'r',label='real')
# plt.legend(loc='upper right')
# plt.xlabel('author number')
# plt.ylabel('papers')
#
# plt.savefig('log/triads/'+str(authorName)+'_MLP.png')
#
# plt.show()
import xgboost as xgb
from xgboost import plot_importance
params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 4,
    'gamma': 0.1,
    'max_depth': 6,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}

plst = params.items()


dtrain = xgb.DMatrix(x_train, y_train)
num_rounds = 500
model = xgb.train(plst, dtrain, num_rounds)

# 对测试集进行预测
dtest = xgb.DMatrix(x_test)
ans = model.predict(dtest)

# 计算准确率
cnt1 = 0
cnt2 = 0
for i in range(len(y_test)):
    if ans[i] == y_test[i]:
        cnt1 += 1

    else:
        cnt2 += 1
print("xgboost classifier:")
print(classification_report(y_test, ans,target_names=target_names))
print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))

# 显示重要特征
plot_importance(model)
plt.savefig('xgboost_imp.pdf')
plt.show()
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(4),
    # SVC(kernel="linear", C=0.025),
    # SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]
for name, clf in zip(names, classifiers):
        # ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(x_train, y_train)
        print(name)

        score = clf.score(x_test, y_test)
        print("score:" + str(score))