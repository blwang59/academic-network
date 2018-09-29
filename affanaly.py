# -*- coding: utf-8 -*-
"""
modified on 2018/07/02

@author: wangbl
Purpose: 分析affs.json的数据。结果为单位变化次数与频率。
待分析数据格式：作者编号：["时间：单位"，“时间：单位”...],发表文章的每一年单位都有记录，但无序。

0703修改：
affs.json经过修改，现在是有序且有每一年单位（只要当年有文章发表）
增加：主题词（这里就是关键词）diversity的计算（也就是不同关键词计数）

0814修改：
增加引用量相关处理

0919修改：
作者涉及主题的相关统计，包括主题分布熵

0921修改
（判断他的主题到底有多发散）、
主题熵随年限的变化趋势（如线性回归后取斜率，或者简单判断是上升中还是下降中还是平稳等）
"""
import json

# affs = json.load(open('E:/WBL/TreeBuilding/data/affs.json'))
#
# affsChangeTimes = {}
# affsChangeFreq = {}
# citations = json.load(open('data/citationsarray_per_author.json'))
#
# maxCitation={}
# for a in citations:
#     maxCitation[a] = max(citations[a])

topics = json.load(open('data/topics.json'))
topicsCount = {}

# for author in topics:
#
#     topicsCount[author] = len(topics[author])
#
#
# fr2 = open('E:/WBL/TreeBuilding/data/topicsCount.json', 'w', encoding='utf-8', errors='ignore')
#
# json.dump(topicsCount, fr2, ensure_ascii='false')


import pandas as pd
import scipy as sc
import scipy.stats
# Input a pandas series
def ent(data):
    p_data= data.value_counts()/len(data) # calculates the probabilities
    entropy=sc.stats.entropy(p_data)  # input probabilities to get the entropy
    return entropy

# for key,value in topics.items():
#     topics[key] = pd.Categorical(value)
for key,value in topics.items():
    # topics[key]=pd.Series(value)
    topics[key]=ent(pd.Series(value))
# print(topics[0])
# topics1 = pd.Categorical(topics)
#     print(topics[key])
print((topics))
fr1 = open('C:/Users/王冰蕾/Documents/academic network/data/topicsDiv.json', 'w', encoding='utf-8', errors='ignore')
json.dump(topics, fr1, ensure_ascii='false')

# 计算单位变化次数########################
# for author in affs:
#     # affsChangeTimes[author] = 0
#     time = []
#     affsOnly = set()
#     # affs[author] = (sorted(set(affs[author])))
#


#     for affrecord in affs[author]:
#
#         time.append(float(affrecord.split(':')[0]))
#         if len(affrecord.split(':')) > 1:
#             affsOnly.add(affrecord.split(':')[1])
#         else:
#             affsOnly.add('-')
#     affsChangeTimes[author] = len(affsOnly)-1
#     # 计算单位变化频率
#     affsChangeFreq[author] = len(affsOnly)-1 / (max(time) - min(time) + 1)
#
# fr1 = open('E:/WBL/TreeBuilding/data/affsChangeTimes.json', 'w', encoding='utf-8', errors='ignore')
# json.dump(affsChangeTimes, fr1, ensure_ascii='false')
#
# fr2 = open('E:/WBL/TreeBuilding/data/affsChangeFreq.json', 'w', encoding='utf-8', errors='ignore')
# json.dump(affsChangeFreq, fr2, ensure_ascii='false')
###############################
