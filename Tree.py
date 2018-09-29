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


first_time = json.load(open('data/first_time.json'))
first_aff = json.load(open('data/first_aff.json'))
author_name = json.load(open('data/name_per_author.json'))
citations = json.load(open('data/citations_per_author.json'))

#hindex = json.load(open('data/hindex.json'))

# seed = '2136372366'#Enhong Chen

# seed = '2153710278' #Hui Xiong
# seed = '2126330539'# Jian Pei
# seed = '2121939561'# Jiawei Han
# seed = '563069026'#Geoffrey E. Hinton
# seed = '161269817'#Yoshua Bengio
seed = '215131072'#Ilya Sutskever
# seed = '2435751034'#Michael I. Jordan



authorName=author_name[seed]
Tree=nx.DiGraph()#Tree是全局变量
Tree.add_node(seed,name = author_name[seed],newRoot = 0)
def onePass():
    '''
    一次扫描数据库
    :return: None
    '''
    with codecs.open('data/data2018_split.csv', 'r', encoding='utf-8', errors='ignore') as f:
       f_csv = csv.reader(f)

       for row in f_csv:
           author = row[3].split(',')
           authors = [item.strip() for item in author] #authors' IDs

           aff = row[2].split(',')
           affs= [item.strip() for item in aff]

           firstAuAff = affs[0]
           firstAu = authors[0]

           topics = row[7]

           for i in range(1,len(authors)):

               if Tree.has_node(authors[i]):#若合作者中有节点存在于树中
                   if Tree.node[authors[i]]['newRoot']==0:
                        Tree.node[authors[i]]['newRoot']=is_new_root(authors[i],affs[i])

                   if firstAuAff==affs[i]:
                           #and first_time[firstAu]>=first_time[authors[i]]:

                       if Tree.has_node(firstAu):#若一作已在树中，仅增加边，不增加节点
                           Tree.add_edge(authors[i], firstAu, topic=topics, time=tell_seniority(authors[i],firstAu),isSameAff=True)
                       else:
                           Tree.add_node(firstAu,name = author_name[firstAu],newRoot = is_new_root(authors[i],affs[i]),citations = citations[firstAu])
                           Tree.add_edge(authors[i],firstAu,topic=topics,time=tell_seniority(authors[i],firstAu),isSameAff=True)
                   else:
                       if Tree.has_node(firstAu):#若一作已在树中，仅增加边，不增加节点
                           Tree.add_edge(authors[i], firstAu, topic=topics, time=tell_seniority(authors[i],firstAu),isSameAff=False)



def tell_seniority(source,dest):
    '''

    :param source:父节点

    :param dest:子节点
    :return:父节点与子节点之间的辈分关系（0同辈，1前辈，-1后辈）
    '''
    if first_time[source]<first_time[dest]-0.5:
        return 1
    elif first_time[source]>first_time[dest]+0.5:
        return -1
    else:
        return 0





def is_new_root(node,aff):
    '''

    :param node: 节点
    :param aff:节点的当前单位
    :return: 此节点是否为一个新的种子节点(是否变换了单位)
    '''

    if first_aff[node] != aff:
        return 1
    else:
        return 0

count = Tree.number_of_nodes()#保存一次扫描数据库之前树的大小，若无增加即为完成
# count=1
onePass()

while count<Tree.number_of_nodes():

    count = Tree.number_of_nodes()
    print(count)
    # print(Tree.number_of_nodes())
    onePass()
    print(Tree.number_of_nodes())

print(len(Tree.nodes()))
print(len(Tree.edges()))
nx.write_gml(Tree,'trees/newdata/'+str(authorName)+'.gml')


