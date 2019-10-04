import math
import json
import numpy as np

Query = [query.strip('\n') for query in open("query_list.txt","r")]
Doc = [doc.strip('\n') for doc in open("doc_list.txt","r")]

query_cnt = len(Query)
doc_cnt = len(Doc)

Param_K1 = 1.6
Param_K3 = 2.4
Param_b = 0.75
Dictionary = {}
DocLen = np.zeros(doc_cnt)

ind = 0
for index, doc in enumerate(Doc):
	for line in open("Document/"+doc,"r").readlines()[3:]:
		line = line.split()[0:-1]
		for word in line:
			DocLen[index] += 1
			if word not in Dictionary:
				Dictionary.update({word:ind})
				ind += 1

for query in Query:
	for line in open("Query/"+query,"r"):
		line = line.split()[0:-1]
		for word in line:
			if word not in Dictionary:
				Dictionary.update({word:ind})
				ind += 1
				
DF = np.zeros(ind)
TF = np.zeros(doc_cnt * ind).reshape((doc_cnt,ind))
TF_ = np.zeros(doc_cnt * ind).reshape((doc_cnt,ind))
QTF = np.zeros(query_cnt * ind).reshape((query_cnt,ind))

for index, doc in enumerate(Doc):
	for line in open("Document/"+doc,"r").readlines()[3:]:
		line = line.split()[0:-1]
		for word in line:
			TF[index][Dictionary[word]] += 1

for index, query in enumerate(Query):
	for line in open("Query/"+query,"r"):
		line = line.split()[0:-1]
		for word in line:
			QTF[index][Dictionary[word]] += 1
			
for index, word in enumerate(Dictionary.keys()):
	df_cnt = 0
	for id, doc in enumerate(Doc):
		if TF[id][index] > 0:
			df_cnt += 1
	DF[index] = df_cnt

for index in range(doc_cnt):
	TF_[index] = np.divide(TF[index],(1 - Param_b + Param_b * DocLen[index] * doc_cnt / np.sum(DocLen)))

f = open("submission.txt", "w")
f.write("Query,RetrievedDocuments\r\n")

for index, query in enumerate(Query):
	f.write(query+",")
	SIM = {}
	s1 = np.zeros(ind)
	s2 = np.zeros(ind)
	s3 = np.zeros(ind)
	for ind, doc in enumerate(Doc):
		s1 = (Param_K1 + 1) * TF_[ind] / (Param_K1 + TF_[ind])
		s2 = (Param_K3 + 1) * QTF[index] / (Param_K3 + QTF[index])
		s3 = np.log(np.divide((doc_cnt - DF + 0.5),(DF + 0.5)))
		s4 = np.sum(s1 * s2 * s3)
		SIM.update({doc:s4})
		
	SIM_SORT = sorted(SIM.items(), key=lambda SIM: SIM[1],reverse=True)
	
	for item in SIM_SORT:
		f.write(item[0] + " ")
	f.write("\r\n")

f.close()