# -*- coding: utf-8 -*-
"""
modified on 2018/06/22

@author: wangbl
Purpose: 将爬到的数据集中需要的部分参数提取出来，作为json文件存储

"""
import codecs
import csv
import json


def add_networks(dic1, key_a, key_b, val):
    if key_a in dic1:
        dic1[key_a].update({key_b: val})
    else:
        dic1.update({key_a: {key_b: val}})


network = {}
ppa = {}
# first_time = json.load(open('E:/WBL/TreeBuilding/data/first_time.json'))
# citations = {}
citations_array ={}
hindex = {}
# last_time = {}
union_ab = {}
time_per_paper={}
paper_per_author = {}
first_aff = {}

citations = json.load(open('data/citationsarray_per_author.json'))
print(len(citations))
# print(len(citations))
for a in citations:
    hindex[a]=sum(i < c for i, c in enumerate(sorted(citations[a], reverse=True)))
#     print(hindex[a])

affs = {}
topics = {}
# only construct v from elder node to yonger node
with codecs.open('E:/WBL/TreeBuilding/data/data2018_split.csv', 'r', encoding='utf-8', errors='ignore') as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        authors = row[3].split(',')

        # citations_paper=row[-1]
        #
        # time = str(row[5].split('-')[0])
        # for a in authors:
        #     if a not in citations_array:
        #         citations_array[a]=[]
        #         citations_array[a].append(int(row[-1]))
        #     else:
        #         citations_array[a].append(int(row[-1]))

        #生成每个节点的单位列表#########################################
        # affs_list = row[2].split(',')
        # k=0
        # for a in authors:
        #     if a not in affs:
        #         affs[a] = []
        #         affs[a].append(str(time)+':'+str(affs_list[k]).strip())
        #         k+=1
        #     else:
        #         addflag=0
        #         for item in affs[a]:
        #             if item.split(':')[0]==time:
        #                 addflag=0
        #             else:
        #                 addflag=1
        #         if addflag==1:
        #             affs[a].append(str(time)+':'+str(affs_list[k]).strip())
        #         k+=1
        ################################

        #生成每个节点的文章关键词列表#########################
        topic_list = list(filter(None,row[7].split(';')))
        # print(set(topic_list))

        for a in authors:
            if a not in topics:
                topics[a] = set()
                if len(topic_list)==1:
                    topics[a].add(str(topic_list))


                else:
                    topics[a].update(set(topic_list))
            else:

                topics[a] = set(topics[a])
                if len(topic_list)==1:
                    topics[a].add(str(topic_list))

                else:

                    topics[a].update(set(topic_list))

        for a in authors:
            # print(topics[a])
            topics[a] = list(topics[a])
            # print(topics[a])



#

        # time_per_paper[row[0]] = time
        # for a in authors:
        #     if a not in first_time:
        #         first_time[a] = time
        #     else:
        #         if time < first_time[a]:
        #             first_time[a] = time
#########
        # for i in range(len(authors)):
        #     # print(authors[i])
        #     # print(first_time.keys())
        # # for a in authors:
        #
        #     if  authors[i] in first_time:
        #         if authors[i] not in first_aff and time == first_time[authors[i]] :
        #             first_aff[authors[i]] = affs[i]
            # else:
            #     if time < first_time[authors[i]]:
            #         first_aff[authors[i]] = affs[i]



        # time = float(row[4].split('-')[0])
        # time_per_paper[row[0]] = time
        # # for a in authors:
        # if authors[0] not in first_time:
        #     first_time[authors[0]] = time
        # else:
        #     if time < first_time[authors[0]]:
        #         first_time[authors[0]] = time
        # for a in authors:
        #     if a == '2136372366':#chenenhong
        #         if '2136372366' not in first_time:
        #             first_time['2136372366'] = time
        #         elif time < first_time['2136372366']:
        #             first_time['2136372366'] = time




    #20180622生成first_time以及time_per_paper(根据data2018生成）
    # fr = open('E:/results/getNewCon/first_time.json', 'w', encoding='utf-8',
    #           errors='ignore')
    # json.dump(first_time, fr, ensure_ascii='false')
    # #
    # fr2 = open('E:/WBL/TreeBuilding/data/time_per_paper.json', 'w', encoding='utf-8',
    #           errors='ignore')
    # json.dump(time_per_paper, fr2, ensure_ascii='false')

    # fr3 = open('E:/WBL/TreeBuilding/data/first_aff.json', 'w', encoding='utf-8',errors='ignore')
    # json.dump(first_aff,fr3,ensure_ascii='false')

    # fr4 =open('E:/WBL/TreeBuilding/data/citations_per_author.json', 'w', encoding='utf-8',errors='ignore')
    # json.dump(citations, fr4, ensure_ascii='false')

    # fr5=open('E:/WBL/TreeBuilding/data/citationsarray_per_author.json', 'w', encoding='utf-8',errors='ignore')
    # json.dump(citations_array, fr5, ensure_ascii='false')
    #fr6的加入需要先跑一遍fr5
    fr6 = open('E:/WBL/TreeBuilding/data/hindex.json', 'w', encoding='utf-8', errors='ignore')
    json.dump(hindex, fr6, ensure_ascii='false')

    # fr7 = open('E:/WBL/TreeBuilding/data/affs.json', 'w', encoding='utf-8', errors='ignore')
    # json.dump(affs, fr7, ensure_ascii='false')

    fr8 = open('E:/WBL/TreeBuilding/data/topics.json', 'w', encoding='utf-8', errors='ignore')
    json.dump(topics, fr8, ensure_ascii='false')

            # if a not in last_time:
            #     2017.0 = time
            # else:
            #     if time > 2017.0:
            #         2017.0 = time

# print('Yanjie Fu:'+str(first_time['2168873515']))
# print('Zijun Yao:'+str(first_time['2229271911']))
# print('Junming Liu:'+str(first_time['2226988312']))
# print('Guannan Liu:'+str(first_time['2273869953']))

#
# with codecs.open('E:/results/getNewCon/data2018.csv', 'r', encoding='utf-8', errors='ignore') as f:
#    f_csv = csv.reader(f)
#    for row in f_csv:
#        authors = row[2].split(',')
#        time = float(row[4].split('-')[0])
#
#
#        for a in authors:
#             if a not in ppa:
#                 ppa[a] = 1
#             else:
#                 ppa[a]+=1

        #20180622加注释
       # for a in authors:
       #     if a not in ppa:
       #         ppa[a] = set(row[0].split())
       #     else:
       #         # weight = (2017.0 - time + 1) / (2017.0 - first_time[a] + 1)
       #         ppa[a].add(row[0])

        #加注释20180622
       # if len(authors) > 1:
       #     for author1 in authors[1:]:
       #     # for author2 in authors:
       #         author2 = authors[0]
       #         if author1 in ppa and author2 in ppa and author1 in first_time and author2 in first_time:
       #             for paper in (ppa[author1] | ppa[author2]):
       #                 if paper in ppa[author2]:
       #                     weight = float(2017.0-time+1)/(2017.0-first_time[author2]+1)
       #                 else:
       #                     weight = float(2017.0 - time + 1) / (2017.0 - first_time[author1] + 1)
       #                 if author1 not in union_ab or author2 not in union_ab[author1]:
       #                     add_networks(union_ab, author1, author2, [weight])
       #                 else:
       #                     union_ab[author1][author2].append(weight)
       #
       #
       #
       #         if author1 in first_time and author2 in first_time and first_time[author1] <= first_time[author2]:
       #             if author1 not in network or author2 not in network[author1]:
       #                 add_networks(network, author1, author2, [float(2017.0-time+1)/(2017.0-first_time[author2]+1)])
       #             else:
       #                 weight = float(2017.0-time+1)/(2017.0-first_time[author2]+1)
       #                 network[author1][author2].append(weight)
   #20180622 生成paper_per_author.json文件(根据data2018生成）
   # fr = open('E:/results/getNewCon/paper_per_author.json', 'w', encoding='utf-8',
   #           errors='ignore')
   # json.dump(ppa, fr, ensure_ascii='false')


                        # elif (2017.0-time+1)/(2017.0-first_time[author2]+1) > network[author1][author2]:
                        #     network[author1][author2] = (2017.0-time+1)/(2017.0-first_time[author2]+1)
#20180622加注释
# for author1 in network:
#    for author2 in network[author1]:
#        network[author1][author2] = sum(network[author1][author2])/sum(union_ab[author1][author2])
#
#
# fr = open('./network_dict_0809_first_site_first_time_with_cheneh.json', 'w', encoding='utf-8', errors='ignore')
# json.dump(network, fr, ensure_ascii='false')



