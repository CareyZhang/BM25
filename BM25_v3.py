import math
import json
import numpy as np

Query = [query.strip('\n') for query in open("query_list.txt","r")]
Doc = [doc.strip('\n') for doc in open("doc_list.txt","r")]

f = open("submission.txt", "w")
f.write("Query,RetrievedDocuments\r\n")

query_cnt = len(Query)
doc_cnt = len(Doc)

Param_K1 = 1.5
Param_K3 = 3.8
Param_b = 0.75
Param_delta = 0.3

for index, query in enumerate(Query):
    f.write(query+",")
    ind = 0
    Dictionary={}
    DocLen = np.zeros(doc_cnt)
    
    for line in open("Query/"+query,"r"):
        line = line.split()[0:-1]
        for word in line:
            if word not in Dictionary:
                Dictionary.update({word:ind})
                ind += 1

    word_cnt = ind
    QTF = np.zeros(word_cnt)
    TF = np.zeros(doc_cnt * word_cnt).reshape((doc_cnt,word_cnt))
    DF = np.zeros(word_cnt)
    
    for line in open("Query/"+query,"r"):
        line = line.split()[0:-1]
        for word in line:
            QTF[Dictionary[word]] += 1
    
    for ind, doc in enumerate(Doc):
        cnt = 0
        for line in open("Document/"+doc,"r").readlines()[3:]:
            line = line.split()[0:-1]
            for word in line:
                DocLen[ind] += 1
                if word in Dictionary.keys():
                    TF[ind][Dictionary[word]] += 1
                    cnt += 1

    for ind, word in enumerate(Dictionary.keys()):
        df_cnt = 0
        for id, doc in enumerate(Doc):
            if TF[id][ind] > 0:
                df_cnt += 1
        DF[ind] = df_cnt
        
    SIM = {}
    s1 = np.zeros(word_cnt)
    s2 = np.zeros(word_cnt)
    s3 = np.zeros(word_cnt)
    for ind, doc in enumerate(Doc):
        s1 = (Param_K1 + 1) * TF[ind] / (Param_K1 * ((1 - Param_b) + Param_b * DocLen[index] * doc_cnt / np.sum(DocLen)) + TF[ind])
        s2 = (Param_K3 + 1) * QTF / (Param_K3 + QTF)
        s3 = np.log(np.divide((doc_cnt - DF + 0.5),(DF + 0.5)))
        s4 = np.sum(s1 * s2 * s3)
        SIM.update({doc:s4})
        
    SIM_SORT = sorted(SIM.items(), key=lambda SIM: SIM[1],reverse=True)
    
    for item in SIM_SORT:
        f.write(item[0] + " ")
    f.write("\r\n")
f.close()