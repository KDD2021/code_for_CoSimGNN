import networkx as nx
import os

# graph_path = "./all/"
# save_path = "./More_than_Ten/"
# c = 0
# for g_file in os.listdir(graph_path):
#     g = nx.read_gexf(graph_path+g_file)
#     if (len(g.nodes())) >= 15:
#     	c += 1
# print(c)
    	# nx.write_gexf(g,save_path+g_file)


save_path = "./More_than_Ten/"

for g_file in os.listdir(save_path):
    g = nx.read_gexf(save_path+g_file)
    print(len(g.nodes()))