import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

path = "./ENZYMES/ENZYMES.edges"


def graph_node_resort(graph):
	g = nx.Graph()
	g_nodes = list(graph.nodes())
	g_edges = list(graph.edges())
	add_edges_list = []
	g.add_nodes_from(list(range(0,len(g_nodes))))
	for u,v,_ in (graph.edges.data()):
		# print(u,v)
	 	add_edges_list.append((g_nodes.index(u),g_nodes.index(v)))
	g.add_edges_from(add_edges_list)
	return g

G=nx.Graph()

with open(path,"r") as f:
	while(1):
		line = (f.readline())
		if not line:
			break
		u, v = line.split(",")
		# count = count-1
		# if count == 0:
		# 	break
		if "\n" in v:
			v = v[:-1]
		# print(u,v)
		G.add_edge(int(u),int(v))
#print(G.nodes())
node_idx = 1
path = "./ENZYMES/ENZYMES.node_attrs"
with open(path,"r") as f:
	while(1):
		line = (f.readline())
		if not line:
			break
		attrs = line.split(",")
		attrs[-1] = attrs[-1][:-1]
		for attr in range(len(attrs)):
			attrs[attr] = float(attrs[attr])

		try:
			nx.set_node_attributes(G,values={node_idx:attrs},name='feat')
			#G.nodes[node_idx]['feat'] = attrs
		except:
			G.add_node(node_idx,feat=attrs,label=node_idx)


		node_idx += 1
print(G.nodes[100]['feat'])
# l = (list(nx.connected_components(G)))
# count = 1
# print(len(l))
# for i in l:
# 	k = G.subgraph(i)
# 	# print(k.edges())
# 	print(k.nodes())
	# kk = graph_node_resort(k)
	# nx.write_gexf(kk,"./all/"+str(count)+".gexf")
	# count=count+1
	# nx.draw(k,pos = nx.spring_layout(k),node_color = 'b',edge_color = 'r',with_labels = False,font_size =18,node_size =3)
	# plt.show()
	# plt.cla()

path = "./ENZYMES/ENZYMES.graph_idx"

with open(path,"r") as f:
	count = 1
	previous_label = 1
	graph_node = []
	node = []
	while(1):
		line = f.readline()
		if not line:
			graph_node.append(node)
			break
		label = int(line)
		if label == previous_label:
			node.append(count)
		else:
			graph_node.append(node)
			node = [count]
		previous_label = label
		count = count+1
# print(len(graph_node))
# print(count)
# print(graph_node[1])
count=1
for i in range(len(graph_node)):
	k = graph_node[i]
	# print(k)
	kk = G.subgraph(k)
	attrs = []
	for node in kk.nodes():
		attrs.append(kk.nodes()[node]['feat'])
	attrs = np.array(attrs)
	kkk = graph_node_resort(kk)

	nx.write_gexf(kkk,"./all/"+str(count)+".gexf",encoding='utf-8')
	p = np.random.random()
	if p < 0.6 :
		nx.write_gexf(kkk, "./train/" + str(count) + ".gexf",encoding='utf-8')
	elif p < 0.8:
		nx.write_gexf(kkk, "./test/" + str(count) + ".gexf",encoding='utf-8')
	else:
		nx.write_gexf(kkk, "./val/" + str(count) + ".gexf",encoding='utf-8')
	np.save("./feature/" + str(count) + ".npy",attrs)
	count = count+1
	# print(kkk.nodes())
	# nx.draw(kkk,pos = nx.spring_layout(kkk),node_color = 'b',edge_color = 'r',with_labels = False,font_size =18,node_size =3)
	# plt.show()
	# plt.cla()