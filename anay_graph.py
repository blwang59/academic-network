# -*- coding: utf-8 -*-
"""
Created on Mon, 2018 Jan 22th 15:30

@author: wangbl
Purpose: new tree construction

"""
import codecs
import csv
import json
import networkx as nx
import pickle
import pandas as pd
import numpy
import os

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
##分析不同的合作模式（同时考虑网络结构和边的标签），对于个人发展（如论文数量、档次等因素）的影响
# g = nx.DiGraph()
# paper_per_author = {}
# with codecs.open('data/test_fullaff.csv', 'r', encoding='utf-8', errors='ignore') as f:
#     f_csv = csv.reader(f)
#     for row in f_csv:
#         author = row[3].split(',')
#         # print('author'+str(author))
#         authors = [item.strip() for item in author]  # authors' IDs
#
#         for a in authors:
#             if a in paper_per_author.keys():
#                 paper_per_author[a]+=1
#             else:
#                 paper_per_author.update({a:1})
#
# fr = open('./data/paper_per_author.json', 'w', encoding='utf-8', errors='ignore')
# json.dump(paper_per_author, fr, ensure_ascii='false')

# authorName='Hui Xiong'
# authorName='Enhong Chen'
# authorName='Jiawei Han'
# authorName='Jian Pei'
from collections import defaultdict
from sklearn import preprocessing

def isSame_degree(G):
    degree = defaultdict(int)
    for u,v in ((u,v) for u,v,d in G.edges(data=True) if d['isSameAff']==1):
        degree[u]+=1
        degree[v]+=1
    return degree


def senority(G):
    '''

    :param G:graph
    :return: 合作人中先辈的数量
    '''
    degree = defaultdict(int)
    for u, v in ((u, v) for u, v, d in G.edges(data=True) if d['time'] == 1):
        degree[u] += 1

        degree[v] += 1
    return degree

def get_features_from_seed(kinds,authorName):
    '''

    :param kinds: 不同的建树方法，可选'term5' or 'with_time_limit'
    :param authorName: 可选的种子节点名称，
    authorName='Hui Xiong'
    authorName='Enhong Chen'
    authorName='Jiawei Han'
    authorName='Jian Pei'
    :return: 相关系数矩阵
    '''

    g = nx.read_gml("./trees/"+str(kinds)+'\/'+str(authorName)+".gml")
    paper_per_author=json.load(open("data/paper_per_author.json"))

    # 边标签：
    # topic time isSameAff


    paper_per_author = json.load(open("data/paper_per_author.json"))
    affsChangeFreq = json.load(open("data/affsChangeFreq.json"))
    affsChangeTimes = json.load(open("data/affsChangeTimes.json"))
    citations = json.load(open("data/citations_per_author.json"))
    hindex = json.load(open("data/hindex.json"))

    # 边标签：
    # topic time isSameAff

    authors_here = {}
    authors_here_affFreq = {}
    authors_here_affTimes = {}
    authors_here_citations = {}
    authors_here_hindex = {}

    for i in paper_per_author:
        if i in g.nodes():
            authors_here.update({str(i): paper_per_author[i]})
    df = pd.DataFrame.from_dict(authors_here, orient='index')

    df.columns = ['papers']
    df.index.names = ['#author']
    # df['#author'].astype('string')
    df['pagerank'] = pd.Series(nx.pagerank(g))



    for i in affsChangeFreq:
        if i in g.nodes():
            authors_here_affFreq.update({str(i):affsChangeFreq[i]})


    for i in affsChangeTimes:
        if i in g.nodes():
            authors_here_affTimes.update({str(i):affsChangeTimes[i]})

    for i in hindex:
        if i in g.nodes():
            authors_here_hindex.update({str(i):hindex[i]})
    for i in citations:
        if i in g.nodes():
            authors_here_citations.update({str(i):citations[i]})

    print(len(hindex))

    df['hindex'] = pd.Series((hindex))

    df['affsChangeTimes'] = pd.Series(affsChangeTimes)
    df['affsChangeFreq'] = pd.Series(affsChangeFreq)
    df['citations'] = pd.Series(citations)

    isSameAff = isSame_degree(g)
    degree = nx.degree(g)

    for a in isSameAff:
        isSameAff[a] = isSameAff[a] / degree[a]

    df['isSameAff'] = pd.Series(isSameAff)

    is_ancestor = senority(g)
    for a in is_ancestor:
        is_ancestor[a] = is_ancestor[a] / degree[a]

    df['isAncestor'] = pd.Series(is_ancestor)

    betweenness = nx.betweenness_centrality(g)
    # print(betweenness)
    # betweenness.update((b,a/((nx.number_of_nodes(g)-1)*(nx.number_of_nodes(g)-2))) for b,a in betweenness.items())
    # df['betweenness'] = [a/((nx.number_of_nodes(g)-1)*(nx.number_of_nodes(g)-2) for a in betweenness.values())]
    df['betweenness'] = pd.Series(betweenness)
    # print(betweenness)

    df = df.fillna(0)


    with open('log/'+str(kinds)+'\/'+str(authorName)+'.txt','w') as f:
        f.write(str(authorName)+'\n')
        f.write('nodes num:' + str(len(g.nodes()))+'\n')
        f.write('edge num:' + str(len(g.edges()))+'\n')
        f.write(str(df.corr()))

    df.to_csv('log/'+str(kinds)+'\/'+str(authorName)+'.csv')

    #normalization
    min_max_scaler = preprocessing.MinMaxScaler()
    pd.DataFrame(min_max_scaler.fit_transform(df), columns=df.columns, index=df.index)
    # df_value = df.values
    # df_normalized = preprocessing.normalize(df_value, norm='l2')
    # df = pd.DataFrame(df_normalized, columns=df.columns)
    df.to_csv('log/'+str(kinds)+'\/'+str(authorName)+'normalized.csv')

    return (df.corr())


# def get_features_from_graph(g):
#     '''
#
#     :param g: 图结构，格式为gml
#     :return: 相关系数矩阵
#     '''
#
#     # g = nx.read_gml("./trees/"+str(kinds)+'\/'+str(authorName)+".gml")
#     paper_per_author=json.load(open("data/paper_per_author.json"))
#     affsChangeFreq = json.load(open("data/affsChangeFreq.json"))
#     affsChangeTimes = json.load(open("data/affsChangeTimes.json"))
#     citations = json.load(open("data/citations_per_author.json"))
#     hindex = json.load(open("data/hindex.json"))
#
#
#     # 边标签：
#     # topic time isSameAff
#
#     authors_here={}
#     authors_here_affFreq={}
#     authors_here_affTimes={}
#     authors_here_citations={}
#     authors_here_hindex={}
#
#     for i in paper_per_author:
#         if i in g.nodes():
#             authors_here.update({str(i):paper_per_author[i]})
#     df = pd.DataFrame.from_dict(authors_here,orient='index')
#     df.columns=['papers']
#     df.index.names = ['#author']
#     df['pagerank']=pd.Series(nx.pagerank(g))
#
#     # for i in affsChangeFreq:
#     #     if i in g.nodes():
#     #         authors_here_affFreq.update({str(i):affsChangeFreq[i]})
#     #
#     #
#     # for i in affsChangeTimes:
#     #     if i in g.nodes():
#     #         authors_here_affTimes.update({str(i):affsChangeTimes[i]})
#     #
#     # for i in hindex:
#     #     if i in g.nodes():
#     #         authors_here_hindex.update({str(i):hindex[i]})
#     # for i in citations:
#     #     if i in g.nodes():
#     #         authors_here_citations.update({str(i):citations[i]})
#     df['hindex'] = df['#author'].map(hindex)
#     df['affsChangeTimes'] = df['#author'].map(affsChangeTimes)
#     df['affsChangeFreq'] =df['#author'].map(affsChangeFreq)
#     df['citations'] = df['#author'].map(citations)
#     isSameAff = isSame_degree(g)
#     degree = nx.degree(g)
#
#     for a in isSameAff:
#         isSameAff[a] = isSameAff[a]/degree[a]
#
#
#     df['isSameAff']=pd.Series(isSameAff)
#
#
#     is_ancestor = senority(g)
#     for a in is_ancestor:
#         is_ancestor[a] = is_ancestor[a]/degree[a]
#
#
#     df['isAncestor'] = pd.Series(is_ancestor)
#     return (df.corr())

def draw_corr(kinds,authorName):
    '''

    :param df:
    :param kinds: 不同的建树方法，可选'term5' or 'with_time_limit'
    :return: 皮尔逊相关系数热力图
    '''
    correlations = get_features_from_seed(kinds,authorName)  #计算变量之间的相关系数矩阵
    # plot correlation matrix
    fig = plt.figure() #调用figure创建一个绘图对象
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)  #绘制热力图，从-1到1
    fig.colorbar(cax)  #将matshow生成热力图设置为颜色渐变条
    ticks = numpy.arange(0,9,1) #生成0-9，步长为1
    ax.set_xticks(ticks)  #生成刻度
    ax.set_yticks(ticks)
    names = ['papers','pagerank',  'hindex','affsChangeTimes','affsChangeFreq','citations','isSameAff',  'isAncestor','betweenness']
    ax.set_xticklabels(names) #生成x轴标签
    ax.set_yticklabels(names)
    plt.savefig('log/'+str(kinds)+'\/'+str(authorName)+'corr.png')
    plt.show()

def draw_corr_from_data(df,kinds,authorName):
    '''

    :param df:
    :param kinds: 不同的建树方法，可选'term5' or 'with_time_limit'
    :return: 皮尔逊相关系数热力图
    '''
    columns = ['papers', 'citations', 'pagerank', 'isSameAff', 'isAncestor', 'betweenness']
    correlations = df[columns].corr()
    print(correlations)#计算变量之间的相关系数矩阵
    # plot correlation matrix
    fig = plt.figure() #调用figure创建一个绘图对象
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)  #绘制热力图，从-1到1
    fig.colorbar(cax)  #将matshow生成热力图设置为颜色渐变条
    ticks = numpy.arange(0,6,1) #生成0-9，步长为1
    ax.set_xticks(ticks)  #生成刻度
    ax.set_yticks(ticks)
    names = ['papers', 'citations', 'pagerank', 'isSameAff', 'isAncestor', 'betweenness']
    ax.set_xticklabels(names) #生成x轴标签
    ax.set_yticklabels(names)
    plt.savefig(check_filename_available('log/'+str(kinds)+'/'+str(authorName)+'corr.png'))
    plt.show()

def check_filename_available(filename):
    n=[0]
    def check_meta(file_name):
        file_name_new=file_name
        if os.path.isfile(file_name):
            file_name_new=file_name[:file_name.rfind('.')]+'_'+str(n[0])+file_name[file_name.rfind('.'):]
            n[0]+=1
        if os.path.isfile(file_name_new):
            file_name_new=check_meta(file_name)
        return file_name_new
    return_name=check_meta(filename)
    return return_name

# threedee = plt.figure().gca(projection='3d')
# threedee.scatter(df['isSameAff'], df['pagerank'], df['papers'])
# threedee.set_xlabel('isSameAff')
# threedee.set_ylabel('pagerank')
# threedee.set_zlabel('papers')
# plt.savefig('log/term5/'+str(authorName)+'3d.png')
# plt.show()

#####################################################
# get_features_from_seed('newdata','Geoffrey E. Hinton')

draw_corr('newdata','Geoffrey E. Hinton')