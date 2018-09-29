# -*- coding: utf-8 -*-

import requests
import json
import re
import codecs
import csv
from time import sleep
import random

def into_csv(data,confname):
    rows = []
    for paper in data["publicationResults"]["publications"]:
        # year = paper["y"]


        authorID = ""
        authorName = ""
        # authorAff = ""
        field = ""
        # title = paper["e"]["dn"]
        if "dn" not in paper["e"]:
            title = "-"
        else:
            title = paper["e"]["dn"]

        if "d" not in paper["e"]:
            abstract = "-"
        else:
            abstract = paper["e"]["d"]

        if "y" not in paper:
            year = "-"
        else:
            year = paper["d"]

        if "cc" not in paper:
            citation = '-'
        else:
            citation = paper['cc']

        if "vfn" not in paper["e"]:
            conference = "-"
        else:
            conference = paper["e"]["vfn"]


        if "aa" not in paper:
            authorID = "-"
            authorName = "-"
        else:
            for author in paper["aa"]:
                authorID += (str(author["auId"]) + ',')
                if "dAfN" not in author:
                    authorName += (author["dAuN"] + ';')
                else:
                    authorName += (author["dAuN"] + '(' + author["dAfN"] + ');')

        if "f" not in paper:
            field = "-"
        else:
            for fields in paper["f"]:
                field += (fields["fn"] + ';')




                # authorAff += (author["dAfN"]+';')

        rows.append({'title': title,
                     'author': authorName,
                     'authorID': authorID[:-1],
                     'abstract': abstract,
                     'time': year,
                     'venue': conference,
                     'field': field,
                     'citation': citation#20180620
                     })

    headers = ['title', 'author', 'authorID',
               'abstract', 'time', 'venue', 'field','citation']
    for i in rows:
        print (i['time'])

    # with codecs.open('./getNewCon/ICDE.csv', 'a+', encoding='utf-8') as f:
#########！！！！
    with codecs.open('./getNewCon/'+confname.upper()+'.csv', 'a+', encoding='utf-8') as f:
        f_csv = csv.DictWriter(f, headers)
        # f_csv.writeheader()
        f_csv.writerows(rows)

def get_total_res(data):
    return int(data['filterGroups']['totalResultCount'])
def main():
    confname = 'sigir'
    # for x in xrange(20, 3600, 20):
    #     data = {'start': '0', 'offset': str(
    # x), '_xsrf': 'a128464ef225a69348cef94c38f4e428'}  #
    # 知乎用offset控制加载的个数，每次响应加载20
    # url2 =""https://academic.microsoft.com/api/search/GetEntityResults?correlationId=ca46ccd7-f644-44d7-9201-ad83fc02c351"
    # url2 = "https://academic.microsoft.com/api/search/GetEntityResults?correlationId=ee5b6b6b-24f0-445a-85ee-20d3cf94d985"#cvpr
    # url2 = "https://academic.microsoft.com/api/search/GetEntityResults?correlationId=e061312c-29b6-4015-ad3e-03f838e54177"#icml
    # url2 = "https://academic.microsoft.com/api/search/GetEntityResults?correlationId=20aeca52-3158-49f3-be34-2bbbede4c3da"#IJCAI
    
    url2 = "https://academic.microsoft.com/api/search/GetEntityResults?correlationId=da5a100f-6397-4986-be15-62b2f21b857f"
    print('now is downloading '+confname)
    #先处理完断掉的年份
    #得到totalres,即每年有多少文章

    brokenyear = 2001 #记录中断的年份
    startyear = 1978 #会议开始的年份
    brokensite=119 #记录中断在5000条时在当年的文章次序
    
    data = {"Query": "And(Composite(C.CN=='"+str(confname)+"'),Y=" + str(brokenyear) + ")",  # cvpr
            # "And(Ty='0', Composite(C.CId=1158167855))", #cvpr
            #         "Query": "And(Ty='0',Composite(C.CId=1203999783))",  # ijcai
            #         "Query": "And(Ty='0',Composite(J.JId = 118988714))",  #jmlr

            "Limit": 8,


            "Offset": 0,
            "OrderBy": "D",
            "SortAscending": False}
    content = requests.post(
        url2, data=data, headers={'X-Requested-With': 'XMLHttpRequest'})  # 用post提交form data
    # print(content.text)
    contents = json.loads('{' + content.text.split('{', 1)[1])
    totalres = get_total_res(contents)
    sleep(random.uniform(0.5, 1))
    #############################################

    for i in range(brokensite, totalres+1, 8):
        data = {"Query":"And(Composite(C.CN=='"+str(confname)+"'),Y="+str(brokenyear)+")",#cvpr
                    # "And(Ty='0', Composite(C.CId=1158167855))", #cvpr
        #         "Query": "And(Ty='0',Composite(C.CId=1203999783))",  # ijcai
        #         "Query": "And(Ty='0',Composite(J.JId = 118988714))",  #jmlr

                "Limit": 8,
                "Offset": str(i),
                "OrderBy": "D",
                "SortAscending": False}
        content = requests.post(
            url2, data=data, headers={'X-Requested-With': 'XMLHttpRequest'}) # 用post提交form data
        # print(content.text)
        contents = json.loads('{'+content.text.split('{',1)[1])
        # print(contents)
        # fr = open('./data.json', 'a+', encoding='utf-8')
        # json.dump(contents, fr, ensure_ascii='false')
        into_csv(contents,confname)

        sleep(random.uniform(0.5, 1))


    ###按年份统计中断年之后的
    for k in range(brokenyear-1,startyear-1,-1):
        #得到totalres,即每年有多少文章
        data = {"Query": "And(Composite(C.CN=='"+str(confname)+"'),Y=" + str(k) + ")",  # cvpr
                # "And(Ty='0', Composite(C.CId=1158167855))", #cvpr
                #         "Query": "And(Ty='0',Composite(C.CId=1203999783))",  # ijcai
                #         "Query": "And(Ty='0',Composite(J.JId = 118988714))",  #jmlr

                "Limit": 8,
                "Offset": 0,
                "OrderBy": "D",
                "SortAscending": False}
        content = requests.post(
            url2, data=data, headers={'X-Requested-With': 'XMLHttpRequest'})  # 用post提交form data
        # print(content.text)
        contents = json.loads('{' + content.text.split('{', 1)[1])
        if contents == '':
            continue
        totalres = get_total_res(contents)
        sleep(random.uniform(0.5, 1))
        #############################################

        for i in range(0, totalres+1, 8):
            data = {"Query":"And(Composite(C.CN=='"+str(confname)+"'),Y="+str(k)+")",#cvpr
                        # "And(Ty='0', Composite(C.CId=1158167855))", #cvpr
            #         "Query": "And(Ty='0',Composite(C.CId=1203999783))",  # ijcai
            #         "Query": "And(Ty='0',Composite(J.JId = 118988714))",  #jmlr

                    "Limit": 8,
                    "Offset": str(i),
                    "OrderBy": "D",
                    "SortAscending": False}
            content = requests.post(
                url2, data=data, headers={'X-Requested-With': 'XMLHttpRequest'}) # 用post提交form data
            # print(content.text)
            contents = json.loads('{'+content.text.split('{',1)[1])
            # print(contents)
            # fr = open('./data.json', 'a+', encoding='utf-8')
            # json.dump(contents, fr, ensure_ascii='false')
            into_csv(contents,confname)

            sleep(random.uniform(0.5, 1))



    # fr = open('./data.json', 'w', encoding='utf-8')
    # json.dump(data, fr, ensure_ascii='false')


if __name__ == '__main__':
    main()
