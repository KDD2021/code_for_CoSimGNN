import csv
import os
import networkx as nx
import numpy as np
from sklearn.metrics import mean_squared_error

def normalize_ds_score(ds_score, g1, g2):
    g1_size = g1.number_of_nodes()
    g2_size = g2.number_of_nodes()
    return np.exp(-(2 * ds_score / (g1_size + g2_size)))

train_path = "../../data/BA_200/test/"
csv_path = "../../data/BA_200/BA_200.csv"

train_graph_set = []

for g_file in os.listdir(train_path):
    g = nx.read_gexf(train_path+g_file)
    g.graph['gid'] = int(g_file.replace(".gexf",""))
    train_graph_set.append(g)

pairs = {}
length = len(train_graph_set)
for sg1 in range(length):
    for sg2 in range(length):
        gid1 = train_graph_set[sg1].graph['gid']
        gid2 = train_graph_set[sg2].graph['gid']
        pairs[gid1,gid2] = (train_graph_set[sg1],train_graph_set[sg2])


csvFile = open(csv_path,"r")
reader = csv.reader(csvFile)

next(reader)
# print(reader)

true_list = []
hungarian_list = []
vj_list = []
beam5_list = []
beam10_list = []

for item in reader:
	
	g1 = int(item[0])
	g2 = int(item[1])
	if (g1,g2) in pairs:
		graph1,graph2 = pairs[g1,g2]

		true = int(item[2])
		hungarian = int(item[3])
		vj = int(item[4])
		beam5 = int(item[5])
		beam10 = int(item[6])

		norm_true = normalize_ds_score(true,graph1,graph2)
		norm_hungarian = normalize_ds_score(hungarian,graph1,graph2)
		norm_vj = normalize_ds_score(vj,graph1,graph2)
		norm_beam5 = normalize_ds_score(beam5,graph1,graph2)
		norm_beam10 = normalize_ds_score(beam10,graph1,graph2)
		
		true_list.append(norm_true)
		hungarian_list.append(norm_hungarian)
		vj_list.append(norm_vj)
		beam5_list.append(norm_beam5)
		beam10_list.append(norm_beam10)


true_list = np.array(true_list)
hungarian_list = np.array(hungarian_list)
vj_list = np.array(vj_list)
beam5_list = np.array(beam5_list)
beam10_list = np.array(beam10_list)

print("hungarian MSE",mean_squared_error(true_list, hungarian_list))
print("vj MSE",mean_squared_error(true_list, vj_list))
print("beam5 MSE",mean_squared_error(true_list, beam5_list))
print("beam10 MSE",mean_squared_error(true_list, beam10_list))

print("hungarian MAE",np.sum(np.absolute((true_list-hungarian_list)))/len(true_list))
print("vj MAE",np.sum(np.absolute((true_list-vj_list)))/len(true_list))
print("beam5 MAE",np.sum(np.absolute((true_list-beam5_list)))/len(true_list))
print("beam10 MAE",np.sum(np.absolute((true_list-beam10_list)))/len(true_list))
    