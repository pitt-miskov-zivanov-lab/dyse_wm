import networkx as nx
from Sensitivity.Pathway_Analysis.YenKShortestPaths import YenKShortestPaths
import matplotlib as mpl
import sys
import matplotlib.pyplot as plt
import queue as Q
import warnings
import re
import csv
import math
sys.setrecursionlimit(15000)

class EXTRACT_PATHS:
    # Every time we visit a node v, we then add a description of all children node of v into a priority queue
    # the format is (priority value: number, description: the list vector of how we reach v)
    # With that, we know what we should expand first. Noe that the description in that priority queue will be automatically sorted by
    # f(v) = h(v) + g(v) where h(v) estimate how close node v is to the target, and g(v) indicate how much cost we've already paid
    def updatePriorityQueue(G, v, pq, attribute):
    	g = 0.0
    	if len(v) != 1:
    		for i in range(len(v)-1):
    			g = g + float(G[v[i]][v[i+1]][attribute])
    	for node in G.successors(v[-1]):
    		g_new = float(G[v[-1]][node][attribute])
    		v.append(node)
    		pq.put(Ordered_partial_path(g+g_new, v))
    		v = v[:-1]
    	return pq

    # Perform expansion at node v, this operation will continue until our indicator reaching_goal is flagged at 1
    # In each expansion, we first check whether node v is target, if not, we add v's chidren node's description to the priority queue(pq)
    # and visit next according to the queue description. Note that we could travel back since some children node may mislead us
    # In addition, we don't allow cycle
    def aStarSearchUtil(G, item, pq, visited, indicator, top_n_paths, source, target, reaching_goal, attribute, top_n, priority_dict, priority_list):
    	if reaching_goal == top_n:
    		return reaching_goal
    	while (reaching_goal != top_n):
    		visited[item[-1]] = True
    		if item[-1] == target:
    			reaching_goal = reaching_goal + 1
    			top_n_paths.append(item)
    			priority_list.append(priority_dict[tuple(item)])
    			item_copy = item
    			for i in list(visited.keys()):
    				visited[i] = False
    			item = [source]
    			pq = Q.PriorityQueue()
    			indicator = False
    		else:
    			pq_list = []
    			pq_index = []
    			pq = EXTRACT_PATHS.updatePriorityQueue(G, item, pq, attribute)
    			for i in range(pq.qsize()):
    				pq_list.append(pq.queue[i].priority)
    				pq_index.append(pq.queue[i].description)
    				priority_dict[tuple(pq_index[-1])] = pq_list[-1]
    			item = pq.get().description
    			while item in top_n_paths:
    				item = pq.get().description

    			if reaching_goal != top_n:
    				reaching_goal = EXTRACT_PATHS.aStarSearchUtil(G, item, pq, visited, indicator, top_n_paths, source, target, reaching_goal, attribute, top_n, priority_dict, priority_list)
    	return reaching_goal

    # Main function and perform the search and plot the path
    def aStarSearch(G, source, target, pos, attribute, top_n=1):
    	visited = {}
    	for node in G.nodes():
    		visited[node] = False
    	top_n_paths = [0,]
    	source_vec = [source]
    	indicator = False
    	pq = Q.PriorityQueue()
    	priority_dict = {}
    	priority_list = []
    	reaching_goal = EXTRACT_PATHS.aStarSearchUtil(G, source_vec, pq, visited, indicator, top_n_paths, source, target, 0, attribute, top_n, priority_dict, priority_list)
    	prev = -1
    	top_n_paths = top_n_paths[1:]
    	for var in top_n_paths[0]:
    		if prev != -1:
    			curr = var
    			#nx.draw_networkx_edges(G, pos, edgelist = [(prev,curr)], width = 2.5, alpha = 0.8, edge_color = 'black')
    			prev = curr
    		else:
    			prev = var
    	return top_n_paths, priority_list

    #takes input from the file and creates a weighted graph
    def CreateGraph(my_dict):

        G = nx.DiGraph()
        for key, value in my_dict.items():
            if isinstance(key, tuple):
                if key[1] != key[0]:
                    if value != 0.0:
                        G.add_edge(key[1], key[0])
                        G[key[1]][key[0]]['standard'] = value
                        G[key[1]][key[0]]['influ'] = -math.log(value)

        for e in G.edges:
            s = 0
            for node in G.predecessors(e[1]):
                s = s + G[node][e[1]]['standard']
                if G[e[0]][e[1]]['standard']/s != 1.0:
                    G[e[0]][e[1]]['sensi'] = -math.log(G[e[0]][e[1]]['standard']/s)
                else:
                    G[e[0]][e[1]]['sensi'] = -math.log(1.0)

        influ = nx.get_edge_attributes(G, 'influ')
        sensi = nx.get_edge_attributes(G, 'sensi')

        return G

    def DrawPath(G, source, target, attribute):
    	warnings.filterwarnings("ignore")
    	pos = nx.spring_layout(G)
    	#val_map = {}
    	#val_map[source] = 'green'
    	#val_map[target] = 'red'
    	#values = [val_map.get(node, 'blue') for node in G.nodes()]
    	#nx.draw(G, pos, with_labels = True, node_color = values, edge_color = 'b' ,width = 1, alpha = 0.7)  #with_labels=true is to show the node number in the output graph
    	#edge_labels = dict([((u, v,), d[attribute]) for u, v, d in G.edges(data = True)])
    	#nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels, label_pos = 0.5, font_size = 11) #prints weight on all the edges
    	return pos

    def YenKSearch(G, source, target, attribute, top_n):
        top_n_paths_result = []
        top_n_paths_cost = []
        ykp = YenKShortestPaths(G, attribute, None)
        top1path = ykp.findFirstShortestPath(source, target)
        if top1path == None:
            return top_n_paths_result, top_n_paths_cost
        else:
            top_n_paths_result.append(top1path.nodeList)
            top_n_paths_cost.append(top1path.cost)
            while top_n > 1:
                topnextpath = ykp.getNextShortestPath()
                if topnextpath == None:
                    return top_n_paths_result, top_n_paths_cost
                top_n_paths_result.append(topnextpath.nodeList)
                top_n_paths_cost.append(topnextpath.cost)
                top_n = top_n -1
            return top_n_paths_result, top_n_paths_cost

class Ordered_partial_path(object):
    def __init__(self, priority, description):
        self.priority = priority
        self.description = description
        return
    def __eq__(self, other):
        return self.priority == other.priority
    def __lt__(self, other):
        return self.priority < other.priority
