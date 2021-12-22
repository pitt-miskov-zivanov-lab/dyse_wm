import sys
import re
import math
from collections import defaultdict

class WDN_Model:
	def __init__(self):
		self.gene_network = defaultdict(list)
		self.alpha_dict = defaultdict(int)
		self.all_elements = set()

	def clear(self):
		self.gene_network.clear()
		self.alpha_dict.clear()
		self.all_elements.clear()

	# static_file is a input stream of a static analyzed file
	def parse_static(self,static_file):

		for line in static_file:
			# Construct edges corresponding to each rule
			if line.find('=')!=-1:
				elements = re.findall(r'\w+',line)
				self.gene_network[elements[0]] = list(set(elements[1:]))
				self.all_elements |= set(elements)
				name = elements[0]
			# Store the activities of each edge
			elif line.find('/')!=-1:
				elements = re.findall(r'\w+',line)
				elements = [e[1:] for e in elements]
				current_tuple = tuple(elements)
			# Store the total activity of a node
			elif line.find('Overall')!=-1:
				current_tuple = (name)
			# Store the probability into the dictionary
			elif line.find('-')!=-1 or re.split(r'\s',line)[0].isdigit():
				if 'influence' not in line:
					occurence = line.count('-')
					self.alpha_dict[current_tuple]+=2**(occurence-len(re.split(r'\s',line)[0]))
				else:
					self.alpha_dict[current_tuple] = float(line.split(' ')[1])

		[self.gene_network[e] for e in self.all_elements]

	# dynamic_file is a input stream of a dynamic analyzed file
	def parse_dynamic(self,dynamic_file):

		for line in dynamic_file:
			# Construct edges corresponding to each rule
			if line.find('=')!=-1:
				elements = re.findall(r'\w+',line)
				self.gene_network[elements[0]] = list(set(elements[1:]))
				self.all_elements |= set(elements)
				name = elements[0]
			# Store the activities of each edge
			elif line.find('/')!=-1:
				elements = re.findall(r'\w+',line)
				elements = [e[1:] for e in elements]
				current_tuple = tuple(elements)
			# Store the total activity of a node
			elif line.find('Overall')!=-1:
				current_tuple = (name)
				self.alpha_dict[current_tuple] = float(re.findall(r'\S+',line)[2])
			# Store the probability into the dictionary
			elif line.find('-')!=-1 or re.split(r'\s',line)[0].isdigit():
				if 'influence' not in line:
					self.alpha_dict[current_tuple]+=float(re.findall(r'\S+',line)[2])
				else:
					self.alpha_dict[current_tuple] = float(line.split(' ')[1])

		[self.gene_network[e] for e in self.all_elements]


	# On courtesy of http://eddmann.com/posts/depth-first-search-and-breadth-first-search-in-python/
	# A generative function returning all possible paths through BFS
	def bfs_paths(self,start,goal):
		stack = [(start, [start])]
		while stack:
			(vertex, path) = stack.pop()
			if len(path) < 11:
				for next in set(self.gene_network[vertex]) - set(path):
					if next == goal:
						yield path + [next]
					else:
						stack.append((next, path + [next]))

		#queue = [(start, [start])]
		#while queue:
			#(vertex, path) = queue.pop(0)
			#for next in set(self.gene_network[vertex]) - set(path):
				#if next == goal:
					#yield path + [next]
				#else:
					#queue.append((next, path + [next]))

	# Find the score of a path, there are several methods to define the score of a path, including naive,
	# normalized, target-biased normalized, delay-free normalized.
	# In default version, we use the naive method, which ends up with calculating the log-probability of a path, i.e., score1
	def path_score(self,path):
		if len(path)<1:
			return float('-inf')

		score1 = 0
		#score2 = 0
		#score3 = 0
		#score4 = 0
		#score5 = 0
		#n1 = 0
		#n2 = 0
		#n3 = 0
		for index in range(len(path)-1):
			# In dynamic analysis, there are some path with zero probability
			if self.alpha_dict[(path[index],path[index+1])] == 0:
				return float('-inf')
			score1 += math.log(self.alpha_dict[(path[index],path[index+1])])
			# score2 += math.log(self.alpha_dict[(path[index],path[index+1])])
			# score3 += math.log(self.alpha_dict[(path[index],path[index+1])])*(len(path)-index-1)
			# n1 += (len(path)-index-1)
			# if (path[index][-2:] != '_D' and path[index][-3:] != '_DD' and path[index][-4:] != '_DDD'):
			# 	score4 += math.log(self.alpha_dict[(path[index],path[index+1])])
			# 	n2 += 1
			# 	score5 += math.log(self.alpha_dict[(path[index],path[index+1])])*(len(path)-index-1)
			# 	n3 += (len(path)-index-1)

		# score2 = score2/len(path)
		# score3 = score3/n1
		# score4 = score4/n2
		# score5 = score5/n3

		return score1

	# Return the path with the highest score
	def optimal_path(self,start,goal):
		max_prob = float('-inf')
		for p in self.bfs_paths(start,goal):
			if self.path_score(p)>max_prob:
				best_path = p
				max_prob = self.path_score(p)

		# Handle the case in which no path exists
		if max_prob == float('-inf'):
			return []

		return best_path


	# Return the top N highest path. If N is not specified, return all paths with score > 0
	def best_N_path(self,start,goal,N=-1,include_score=0):
		all_path = []
		for p in self.bfs_paths(start,goal):
			all_path.append((self.path_score(p),p))

		# all_path variable includes score and path
		all_path = sorted(all_path,key=lambda x:-x[0])

		# Output format handling
		if N != -1:
			if include_score:
				return [(k,v) for k,v in all_path[:N] if k != float('-inf')]
			else:
				return [v for k,v in all_path[:N] if k != float('-inf')]
		else:
			if include_score:
				return [(k,v) for k,v in all_path if k != float('-inf')]
			else:
				return [v for k,v in all_path if k != float('-inf')]


	# Return the nodes' importance in the input/output path
	def importance_in_path(self,start,goal,node_of_interest):
		#s = self.logadd([k for k,v in self.best_N_path(start,goal,-1,1) if node_of_interest in v])
		#return math.exp(s)
		res = [float('-inf') for index in range(len(node_of_interest))]
		for k,v in self.best_N_path(start,goal,-1,1):
			for index in range(len(node_of_interest)):
				# If the node is in the path, add k to the score of the node
				if node_of_interest[index] in v:
					res[index] = self.logadd([res[index],k])

		res = [math.exp(e) for e in res]
		return res

	# Return the top N important node in the input/output path. Input/output are excluded
	def best_N_importance_in_path(self,start,goal,N=-1,include_score=0):
		score = self.importance_in_path(start,goal,sorted(self.all_elements))
		score_name = list(zip(score, sorted(self.all_elements)))
		score_name = sorted(score_name,key=lambda x:-x[0])

		# Return everything if number not specified
		if N == -1 or N > len(self.all_elements):
			remained = len(self.all_elements) + 1
		else:
			remained = N

		index = 0
		res = list()
		while remained > 0 and index < len(self.all_elements):
			# Iterate until we have enough or exceed number of elements
			if score_name[index][1] != start and score_name[index][1] != goal:
				if include_score:
					res.append(score_name[index])
				else:
					res.append(score_name[index][1])
				remained -= 1
			index += 1

		return res


	# Calculate log(exp x + exp y) without taking the exponential
	def logadd(self,elements):
		if len(elements)==0:
			return float("-inf")

		if len(set(elements)) <= 1 and elements[0]==float("-inf"):
			return float("-inf")

		result = elements[0]
		for i in range(1,len(elements)):
			if result==float("-inf") and elements[i]==float("-inf"):
				continue
			elif elements[i] <= result:
				result += math.log(1+math.exp(elements[i]-result))
			else:
				result = elements[i] + math.log(1+math.exp(result-elements[i]))

		return result

	# Output link number matrix to a stream file
	def link_number_matrix(self,op):

		for key1 in sorted(self.gene_network):
			lengths = []
			for key2 in sorted(self.gene_network):
				lengths += [str(len([p for p in self.bfs_paths(key1,key2)]))]
			op.write(','.join(lengths))
			op.write('\n')

	# Functions for output use (to return all results)

	# return the element list in A-Z order
	def element_list(self):
		return [key for key in sorted(self.gene_network)]

	# return the element sensitivity in a list with A-Z order
	def element_sensitivity_list(self):
		res = list()
		for key1 in sorted(self.gene_network):
			alpha = 0
			for key2 in sorted(self.gene_network):
				alpha += self.alpha_dict[(key1,key2)]
			res += [alpha]
		return res

	# Return the immediate influence matrix
	def immediate_influence_matrix(self):
		index = sorted(self.gene_network)
		res = [[0.0 for i in range(len(index))] for i in range(len(index))]
		for i in range(len(index)):
			for j in range(len(index)):
				if self.alpha_dict[(index[i],index[j])] != 0:
					res[j][i] = self.alpha_dict[(index[i],index[j])]
		return res

	# Return the element to element influence matrix
	def ETEI_matrix(self, status_report=False):
		res = list()
		size = len(self.gene_network) * len(self.gene_network)
		status = 0
		if status_report:
			print("Generating Matrix:\n")
		for key1 in sorted(self.gene_network):
			scores = []
			for key2 in sorted(self.gene_network):
				status = status + 1
				if status_report:
					print("{:.3%}\n".format(status/size))
				scores += [math.exp(self.logadd([k for k,v in self.best_N_path(key2,key1,-1,1)]))]
			res.append(scores)
		return res

	# Return the element to element pathway number matrix
	def ETEPnumber_matrix(self):
		res = list()
		for key1 in sorted(self.gene_network):
			lengths = []
			for key2 in sorted(self.gene_network):
				lengths += [len([p for p in self.bfs_paths(key1,key2)])]
			res.append(lengths)
		return res

	# Return the element change probability, for boolean model only
	def element_change_pro_list(self):
		return [self.alpha_dict[(key)] for key in sorted(self.gene_network)]


	# Functions for output use (to return inquired results)
	# return the sensitivity of the inq_ele
	def element_sensitivity(self, inq_ele):
		res = 0.0
		for key in sorted(self.gene_network):
			tuple = (inq_ele,key)
			if tuple in self.alpha_dict:
				res += self.alpha_dict[tuple]
		return res

	# return the immediate influence from element1 to element2
	def element_immediate_influence_from_to(self, element1, element2):
		if self.alpha_dict[(element2,element1)] != 0:
			return self.alpha_dict[(element2,element1)]
		else:
			return 0.0

	# return the immediate influce of the inquiry element on other elements in a dictionary
	def element_immediate_influence_from(self, inq_ele):
		keys = [key for key in sorted(self.gene_network)]
		values = [self.alpha_dict[(key,inq_ele)] for key in keys]
		d = dict(zip(keys, values))
		d = {key:val for key, val in d.items() if val != 0.0}
		return sorted(d.items(), key=lambda x: x[1], reverse=True)

	# return the immediate influce of other elements on the inquiry element
	def element_immediate_influence_to(self, inq_ele):
		keys = [key for key in sorted(self.gene_network)]
		values = [self.alpha_dict[(inq_ele, key)] for key in keys]
		d = dict(zip(keys, values))
		d = {key:val for key, val in d.items() if val != 0.0}
		return sorted(d.items(), key=lambda x: x[1], reverse=True)

	# return the ETEI from element1 to element2
	def ETEI_from_to(self, element1, element2):
		return math.exp(self.logadd([k for k,v in self.best_N_path(element2,element1,-1,1)]))

	# return the ETEI of the inquiry element on other elements
	def ETEI_from(self, inq_ele):
		keys = [key for key in sorted(self.gene_network)]
		values = [math.exp(self.logadd([k for k,v in self.best_N_path(key,inq_ele,-1,1)])) for key in keys]
		d = dict(zip(keys, values))
		d = {key:val for key, val in d.items() if val != 0.0}
		return sorted(d.items(), key=lambda x: x[1], reverse=True)

	# return the ETEI of other elements on the inquiry element
	def ETEI_to(self, inq_ele):
		keys = [key for key in sorted(self.gene_network)]
		values = [math.exp(self.logadd([k for k,v in self.best_N_path(inq_ele,key,-1,1)])) for key in keys]
		d = dict(zip(keys, values))
		d = {key:val for key, val in d.items() if val != 0.0}
		return sorted(d.items(), key=lambda x: x[1], reverse=True)
