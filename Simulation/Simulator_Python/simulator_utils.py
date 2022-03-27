import numpy as np
import pandas as pd
import json
import os
import re
import matplotlib.pyplot as plt
from Simulation.Simulator_Python import simulator_interface as sim
from Visualization import visualization_interface as viz
from Translation.model import get_model, get_model_template, get_model_from_delphi
import seaborn as sns
from scipy import signal, interpolate
from scipy.optimize import nnls
from statsmodels import api as sm
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error
from collections import defaultdict

def lag_calc(a,b):
	"""Finds discrete delay between time series a and b."""
	correlation = signal.correlate(a,b, mode="full")
	return np.nanargmax(correlation)+1-len(a)

def date2step(target_date:str, start_date:str, end_date:str, num_steps:int):
	"""Converts target_date to simulation time step, given start_date and end_date"""
	try:
		start = pd.to_datetime(start_date)
	except:
		raise ValueError('Start date not properly formated. Try string in format "YYYY-MM-DD".')
	try:
		end = pd.to_datetime(end_date)
	except:
		raise ValueError('End date not properly formated. Try string in format "YYYY-MM-DD".')
	try:
		target = pd.to_datetime(target_date)
	except:
		raise ValueError('Target date not properly formated. Try string in format "YYYY-MM-DD".')

	assert end>=start, "end_date needs to be at a later date than start_date."
	assert target>=start, "Target date needs to be at a later date than start_date."
	assert target<=end, "Target date needs to be at an earlier date than end_date."

	return int(((target-start)/(end-start))*num_steps)

def step2date(target_step:int, start_date:str, end_date:str, num_steps:int):
	"""Converts target_step to date, given start_date and end_date"""
	try:
		start = pd.to_datetime(start_date)
	except:
		raise ValueError('Start date not properly formated. Try string in format "YYYY-MM-DD".')
	try:
		end = pd.to_datetime(end_date)
	except:
		raise ValueError('End date not properly formated. Try string in format "YYYY-MM-DD".')

	return start+(target_step/num_steps)*(end-start)

def parse_nodes_and_edges(model_spreadsheet:str='model.xlsx'):
	"""Extract nodes and edges information from a model spreadsheet"""
	df_default = pd.read_excel(model_spreadsheet).fillna('')
	edges = dict()
	elements = df_default['Element Name'].values
	for i, it in df_default.iterrows():
		reg = it['Positive']
		L = reg.split('+')
		if L!='':
			L = reg.split('+')
			for l in L:
				if not l.startswith('-') and l!='':
					weight, reg = l.split('*')
					wt, wl = weight.split('&')
					edges[(reg,it['Element Name'])] = {'source':reg, 'target':it['Element Name'],
										 'trend-weight':float(wt), 'level-weight':float(wl), 'polarity':+1}
		reg = it['Negative']
		L = reg.split('+')
		if L!='':
			for l in L:
				if not l.startswith('-') and l!='':
					weight, reg = l.split('*')
					wt, wl = weight.split('&')
					edges[(reg,it['Element Name'])] = {'source':reg, 'target':it['Element Name'],
										 'trend-weight':float(wt), 'level-weight':float(wl), 'polarity':-1}
	return elements, edges

def update_model(original_model:str, updated_model:str, \
				 edges:dict):
	"""This function updates edges information in the model spreadsheet and stores it under
	the new designated name updated_model."""
	assert updated_model.endswith('.xlsx'), "Argument 'updated_model' needs to have extension '.xlsx'."
	df = pd.read_excel(original_model).fillna('')
	id2name = dict(zip(df['Element IDs'], df['Variable']))
	for i, cur_elem in enumerate(df['Element IDs']):
		pos_regs = ["{0}&{1}*{2}".format(v['trend-weight'],v['level-weight'],id2name[v['source']]) \
						for v in edges.values() if v['target']==cur_elem and v['polarity']==1]
		neg_regs = ["{0}&{1}*{2}".format(v['trend-weight'],v['level-weight'],id2name[v['source']]) \
						for v in edges.values() if v['target']==cur_elem and v['polarity']==-1]
		pos_counter_terms = ["-{0}&0*{2}".format(v['trend-weight'],v['level-weight'],id2name[v['source']]) \
						for v in edges.values() if v['target']==cur_elem and v['polarity']==-1]
		neg_counter_terms = ["-{0}&0*{2}".format(v['trend-weight'],v['level-weight'],id2name[v['source']]) \
						for v in edges.values() if v['target']==cur_elem and v['polarity']==1]
		pos_reg_string = "+".join(pos_regs+pos_counter_terms)
		neg_reg_string = "+".join(neg_regs+neg_counter_terms)
		df.loc[i,'Positive'] = pos_reg_string
		df.loc[i,'Negative'] = neg_reg_string
	df.to_excel(updated_model,index=False)

def decode_DySE(val_perc, stdev_perc, lower_bound, upper_bound, joint_TL, max_level:int=30):
	"""Interprets DySE simulation results according to the context of historical data.
	
	"""
	df_dyn_proj = pd.DataFrame()
	LB, UB, val_perc, stdev_perc = np.array(lower_bound), np.array(upper_bound), np.array(val_perc), np.array(stdev_perc)
	df_dyn_proj['value'] = LB + val_perc*(UB-LB)/100.
	df_dyn_proj['lower'] = np.array(df_dyn_proj['value']) - stdev_perc*(UB-LB)/100.
	df_dyn_proj['upper'] = np.array(df_dyn_proj['value']) + stdev_perc*(UB-LB)/100.
	df_dyn_proj.index = joint_TL
	return df_dyn_proj

def find_heads(default_model:str):
	"""Finds head nodes in the DySE model spreadsheet"""
	df_default = pd.read_excel(default_model).fillna('')
	assert 'Positive' in df_default.columns, "Positive column is missing"
	assert 'Negative' in df_default.columns, "Negative column is missing"
	element_ID_col = re.findall(r'Element ID[s]?', ';'.join(df_default.columns.tolist()))
	assert len(element_ID_col)>=1, "Column Element IDs is missing"
	el_ID_col = element_ID_col[0]
	return df_default[(df_default['Positive']=='')&(df_default['Negative']=='')][el_ID_col].tolist()

def calibrate(X, lower_bound=None, upper_bound=None, start:int=0, end:int=-1,  
			  levels:int=3, initial_trend:float=0, top_padding:int=0, bottom_padding:int=0):
	"""This function discretizes a time series, calculates trends, discretizes trends,
	and stores relevant calibration parameters.
	
	Parameters:
	- X (iterable): time series 
	- lower_bound (float): lower bound for the value range
	- upper_bound (float): upper bound for the value range
	- start (int): start index for the desired slice of X (default: 0)
	- end (int): end index for the desired slice of X (default: -1)
	- levels (int): number of discrete levels in the historical data range (default: 3)
	- initial_trend (float): initial trend of the element, default valu (default: 0)
	- top_padding (int): number of discrete levels above the historical range (default: 0)
	- bottom_padding (int): number of discrete levels below the historical range (default: 0)
	
	Returns:
	- series_refined (dict): a dictionary containing original time series, discretized slice,
	normalized time series, normalized discrete time series, trends slice, discretized trends slice,
	normalized trends, normalized discrete trends, max level, lower bound, upper bound, bin size,
	bottom padding, and top padding."""
	
	if lower_bound!=None:
		lb = lower_bound
	else:
		lb = np.nanmin(X)
	if upper_bound!=None:
		ub = upper_bound
	else:
		ub = np.nanmax(X)
	MAX_LEVEL = bottom_padding+levels+top_padding-1
	# Routine checks
	if end==-1:
		end = len(X)
	assert end>start, "End (index) needs to be greater than start (index)."
	assert start>=0 and int(start)==start, "Start needs to be nonnegative index"
	assert end>=0 and int(end)==end, "Start needs to be nonnegative index"
	assert end-start<=len(X), "Slice cannot be longer than the input series"
	
	# Slice the [start:end] interval
	xdata = np.array(X[start:end])
	assert not np.any(pd.isna(xdata)==True), "NaN value present in the discretized span"
	assert levels>0 and levels==int(levels), "levels needs to be nonzero integer value"
	
	if levels>1:
		binsize = (ub-lb)/float(int(levels)-1)
		disc_data = np.ceil((xdata - lb)/binsize) + bottom_padding
	elif levels==1:
		binsize = 0
		disc_data = np.zeros_like(xdata).astype(int) + bottom_padding
	else:
		binsize = -1
	
	# Normalize to [0,1]
	xdata_normal = (xdata-lb) / float(ub-lb)
	disc_data_normal = disc_data / float(MAX_LEVEL)
	
	# Trend initialization
	if start==0:
		init_trend = initial_trend
	elif np.isnan(X[start-1]):
		init_trend = initial_trend
	else:
		init_trend = X[start] - X[start-1]
	# Calculating trends        
	xtrend = np.concatenate([np.ones(1)*init_trend,xdata[1:]-xdata[:-1]])
	
	# Trend discretization
	disc_init_trend = np.sign(init_trend)*np.ceil(np.abs(init_trend)*MAX_LEVEL)
	disc_trend = np.concatenate([np.ones(1)*disc_init_trend, disc_data[1:]-disc_data[:-1]])
		
	# Normalize trends to [-1,1]
	xtrend_normal = xtrend / float(MAX_LEVEL)
	disc_trend_normal = disc_trend / float(MAX_LEVEL)
	
	series_refined = {'data':xdata, 'discrete_data':disc_data, 
					  'normalized_data':xdata_normal, 'normalized_discrete_data': disc_data_normal,
					 'trends':xtrend, 'discrete_trends':disc_trend, 
					  'normalized_trends':xtrend_normal, 'normalized_discrete_trends': disc_trend_normal,
					 'max_level':MAX_LEVEL, 'lower_bound':lb, 'upper_bound':ub, 'bin_size':binsize,
					 'bottom_padding':bottom_padding, 'top_padding':top_padding}
	return series_refined

def fill_nan(A):
	'''
	interpolate to fill nan values
	'''
	if len(A)<=1:
		return A
	inds = np.arange(A.shape[0])
	good = np.where(np.isfinite(A))
	f = interpolate.interp1d(inds[good], A[good],bounds_error=False)
	B = np.where(np.isfinite(A),A,f(inds))
	return B

def array2toggle_string(a, scale:int=1):
	"""This function converts a list of values to DySE value toggling notation.
	For toggling at every n-th time step, set scale=n."""
	try:
		a = np.array(a,np.int16)
		return ",".join([str(a[0])]+["{0}[{1}]".format(x,(i+1)*scale) for i,x in enumerate(a[1:])])
	except:
		raise TypeError('Input a needs to be a sequence (list, array or tuple)!')
	return ''

def build_toggle_string(a, minVal:float, maxVal:float, levels:int=30, max_level:int=30,
					offset:int=1, scale:int=1, default_string:str='15', bottom_padding:int=10,
					timesteps=None):
	"""This function is converting the constrained values into DySE initialization strings."""
	
	if a is None or len(a)==0:
		return default_string

	try:
		a = np.array(a,np.int16)
	except:
		raise TypeError('Input a needs to be a sequence (list, array or tuple) of integers!')
	
	if not timesteps is None:
		a = a[:len(timesteps)]
		if timesteps[0]==0:
			return ",".join([str(a[0])]+["{0}[{1}]".format(x,i) for i,x in zip(timesteps[1:],a[1:])])
		else:
			return 

	return ",".join([str(a[0])]+["{0}[{1}]".format(x,i*scale+offset) for i,x in enumerate(a[1:])])

def initialize_model(default_model:str,
					output_model:str,
					model_json:str,
					experiment_json:str,
					time_units:float=None,
					visualize:bool=False, 
					max_level:int=30,
					scale_factor:float=1.,
					sim_scheme:str='seq'):
	"""This function takes a raw model, and sets initial values according to the historical data.

	Inputs:
	- default_model [str]: name (path) to the starting Excel model spreadsheet
	- output_model [str]: name (path) for the initialized model spreadsheet
	- model_json [str]: name (path) for the JSON file with CauseMos model dump, containing
						historical data
	- experiment_json [str]: name (path) for the JSON file with CauseMos experiment request, containing
						projection start and end dates
	- time_units [int]: number of time points that the user would like to see projections displayed for
	- visualize [bool]: flag for whether to visualize static baseline projections
	- max_level [int]: default value of maximum discrete level
	- scale_factor [float]: scales the number of simulation steps to tune temporal granularity
							(default: 1.0)
	- sim_scheme [str]: simulation scheme, choice between 'seq' (sequential, random asynchronous) and 'sim'
						(simultaneous), which defines updates ordering in simulation (default: 'seq')"""
	# load the historical data
	with open(model_json,'r') as jf:
		dump = json.load(jf)
	CM_data = CauseMosIndicators(dump['id'])
	CM_data.load_data(data_json=model_json, request_json=experiment_json)
	with open(experiment_json,'r') as expt:
		expt_info = json.load(expt)

	# Extract bounds
	lower_bounds = {k:CM_data.indicators[k]['minValue'] \
					for k in CM_data.indicators if 'minValue' in CM_data.indicators[k]}
	upper_bounds = {k:CM_data.indicators[k]['maxValue'] \
					for k in CM_data.indicators if 'maxValue' in CM_data.indicators[k]}

	# Extract max_level:
	max_level = np.max([CM_data.indicators[k]['numLevels']-1 \
					for k in CM_data.indicators if 'numLevels' in CM_data.indicators[k]])

	assert expt_info["experimentType"]=='PROJECTION', "Experiment type needs to be 'PROJECTION' to proceed with simulation."

	# Extract projection start and end dates
	if "experimentParam" in expt_info:
		expParam = "experimentParam"
	elif "experimentParams" in expt_info:
		expParam = "experimentParams"
	else:
		raise ValueError("experimentParams key not found in the experiment request json!")

	assert "startTime" in expt_info[expParam], "Missing 'startTime' from the projection request JSON"
	assert "endTime" in expt_info[expParam], "Missing 'endTime' from the projection request JSON"
	projection_start = pd.Timestamp(expt_info[expParam]["startTime"], unit='ms').strftime('%Y-%m-%d')
	projection_end = pd.Timestamp(expt_info[expParam]["endTime"], unit='ms').strftime('%Y-%m-%d')
	
	# Extract number of steps
	if time_units is None:
		if 'numTimesteps' in expt_info[expParam]:
			time_units = expt_info[expParam]['numTimesteps']
		elif 'numTimeSteps' in expt_info[expParam]:
			time_units = expt_info[expParam]['numTimeSteps']
		else:
			raise ValueError("numTimesteps is missing from experiment JSON")
	

	# build the common timeline for all the elements and make baseline projections from the historical data
	JH = CM_data.joint_history(elements=list(CM_data.historical_data.keys()), combination_method='union')
	if not JH is None:
		CM_data.baseline_projection(data=CM_data.indicators,
									time_series=CM_data.time_series, 
									history_start_date=JH[0], 
									history_end_date=JH[-1],
									#history_end_date=np.nanmin([JH[-1], pd.Timestamp(projection_start)]), 
									history_timelines=CM_data.history_timelines,
									projection_start_date=projection_start,
									projection_end_date=projection_end,
									freq=CM_data.frequencies,
									out_path='.',
									visualize=visualize,
									verbose=False)
	
	# parse nodes and edges from the spreadsheet
	df_model = pd.read_excel(default_model).fillna('')
	nodes, edges = parse_nodes_and_edges(default_model)
	df_edges = pd.DataFrame.from_dict(edges).T
	df_edges.index = range(df_edges.shape[0])
	if 'Scenario' in ';'.join(list(df_model.columns)):
		scenario_vs_initial = 'Scenario'
	else:
		scenario_vs_initial = 'Initial'

	# find the head nodes
	CM_data.head_nodes = find_heads(default_model=default_model)
	
	# Using right column name for Element ID (either 'Element ID' or 'Element IDs')
	element_ID_col = re.findall(r'Element ID[s]?', ';'.join(df_model.columns.tolist()))
	assert len(element_ID_col)>=1, "Column Element IDs is missing"
	el_ID_col = element_ID_col[0]
	
	# Baseline projection number of points
	num_points = len(CM_data.joint_projection_timeline)

	# Number of simulation time steps
	num_steps = sim.calculate_simulation_steps( inputFilename=default_model, 
												time_units=time_units, 
												scale_factor=scale_factor )
	if num_steps == 0:
		num_steps = time_units

	# Number of time steps during a single period between two consecutive time points 
	# (e.g. if monthly resolution, how many simulation time steps correspond to one month)
	if sim_scheme=='seq':
		cycle_length = num_steps//time_units
	else:
		cycle_length = 1

	projection_timesteps = [j*cycle_length for j in range(len(CM_data.joint_projection_timeline)+1)]

	# A dictionary for storing aggregated constraints for each element
	cn_agg_means = dict()

	for i, element in enumerate(df_model[el_ID_col]):
		# set default levels (actually max discrete level) and initial level (mid level)
		levels = max_level
		initial_string = str(max_level//2)
		# Check if the element has historical data - if it doesn't, use default initial value
		if (not element in CM_data.historical_data) or \
			(CM_data.historical_data[element]['values'] is None) or \
			JH is None: # if there's no joint history (i.e. no elements have historical data)
			df_model.loc[i,scenario_vs_initial+' 0'] = initial_string
			continue

		# store historical data for more convenient use
		if element in CM_data.historical_data:
			if 'values' in CM_data.historical_data[element]:
				hist_data = CM_data.historical_data[element]['values']
			else:
				hist_data = {'values':None}
		else:
			hist_data = {'values':None}
		if element in lower_bounds:
			lb = lower_bounds[element]
			bottom_padding = 0
		else:
			lb = np.nanmin(hist_data)
			bottom_padding = 10
			levels -= 10
		if element in upper_bounds:
			ub = upper_bounds[element]
			top_padding = 0
		else:
			ub = np.nanmax(hist_data)
			top_padding = 10
			levels -= 10
		
		# Slice the projections
		data_array = CM_data.stat_forecast_slice[element]['mean'].values
		#data_array = CM_data.baseline_projections[element][projection_start:projection_end]['mean'].values
		
		# Aggregate data if resolution is too granular
		timesteps_slice = projection_timesteps[:len(data_array)]

		#data_sliced = data_array[:len(projection_timesteps)]
		data_sliced = data_array[:len(timesteps_slice)]

		data_aggregation = {s:[] for s in sorted(set(timesteps_slice))}
		for s,v in zip(timesteps_slice, data_sliced):
			data_aggregation[s].append(v)
		data_means = {j:np.nanmean(v) for j,v in data_aggregation.items() if len(v)>0}

		# Initialize clamp Boolean variable to mark if the element has been clamped for projections
		clamped = False
		for cnstr in expt_info[expParam]['constraints']:
			cn_agg = {j:[] for j in range(num_steps)}
			if cnstr['concept']==element:
				# Check if clamped
				if len(cnstr['values'])>0:
					clamped = True
				for c1 in cnstr['values']:
					if c1['step']*cycle_length<num_steps:
						cn_agg[c1['step']*cycle_length].append(c1['value'])

			cn_agg_means[element] = {j:np.nanmean(v) for j,v in cn_agg.items() if len(v)>0}
			if clamped and len(cn_agg_means[element])>0:
				clamp_start = np.nanmin(list(cn_agg_means[element].keys()))
			else:
				clamp_start = np.nan
			if not np.isnan(clamp_start):
				data_means = {k:v for k,v in data_means.items() if k<clamp_start}
			for j, v in cn_agg_means[element].items():
				data_means[j] = v

		
		data_array_reduced = np.array([v for s,v in sorted(data_means.items(), key=lambda x: x[0])])
		timesteps_reduced = np.array([s for s,v in sorted(data_means.items(), key=lambda x: x[0])])

		# discretize
		discrete_array = np.ceil((data_array_reduced-lb)*levels/(ub-lb)) + bottom_padding
		discrete_array = np.where(discrete_array>max_level, max_level, discrete_array)
		discrete_array = np.where(discrete_array<0, 0, discrete_array)

		if element in CM_data.head_nodes:
			freq = CM_data.frequencies[element]
			if freq=='year':
				try:
					offset = np.where(CM_data.joint_projection_timeline.year>CM_data.joint_projection_timeline[0].year)[0][0]
				except:
					offset = 12
			elif freq=='month':
				offset = 1
			else:
				raise ValueError('Unrecognized temporal resolution: {0}. It needs to be either "year" or "month"'.format(freq))
			
			initial_string = build_toggle_string(a=discrete_array, minVal=lb, maxVal=ub, 
												levels=levels, max_level=max_level,
												bottom_padding=bottom_padding, 
												offset=offset, timesteps=timesteps_reduced)
			
			df_model.loc[i,scenario_vs_initial+' 0'] = initial_string

		else:
			if element in cn_agg_means:
				# toggle_pairs = []
				# for s,v in cn_agg_means[element].items():
				# 	if s==0:
				# 		continue
				# 	toggle_value = int(np.max([np.min([np.ceil((v-lb)*levels/(ub-lb))+bottom_padding,max_level]),0]))
				# 	toggle_pairs.append((s,toggle_value))
				toggle_pairs = [(s,int(np.max([np.min([np.ceil((v-lb)*levels/(ub-lb))+bottom_padding,max_level]),0]))) \
					for s,v in cn_agg_means[element].items() if s!=0]
				toggle_string_list = ["{0}[{1}]".format(v,s) for s,v in sorted(toggle_pairs, key=lambda x: x[0])]
			else:
				toggle_string_list=[]
			try:
				starting_level = str(int(discrete_array[0]))
			except:
				starting_level = str(int(max_level//2))
			initial_string = ','.join([starting_level]+toggle_string_list)
			df_model.loc[i,scenario_vs_initial+' 0'] = initial_string

	df_model.to_excel(output_model, index=False)

def cut_NaNs(M, axis=0):
	"""Return matrix with rows containing NaNs sliced out."""
	idx = []
	for i in range(M.shape[axis]):
		if (axis==0 and not np.any(pd.isna(M[i,:]))) or (axis==1 and not np.any(pd.isna(M[:,i]))):
			idx.append(i)
	if axis==0:
		return M[idx,:]
	elif axis==1:
		return M[:,idx]

def discretize(a, data=None, levels=31, lb=None, ub=None, b_pad=10, t_pad=10, normalize:bool=True):
	"""Convert an array into its discretized DySE variables. Set normalize=True to normalize to
	interval [0,1]."""
	a = np.array(a)
	if np.all(pd.isna(a)):
		return a
	if data is None:
		data = a
	min_val = np.nanmin(data)
	max_val = np.nanmax(data)
	if lb is None:
		lb = 2*min_val-max_val
	if ub is None:
		ub = 2*max_val-min_val
	MAX_LVL = levels-1
	
	if normalize:
		return np.ceil((a-lb)*MAX_LVL/(ub-lb))/MAX_LVL
	else:
		return np.ceil((a-lb)*MAX_LVL/(ub-lb))

def build_df_data(indicator_data:dict):
	"""This function builds a table with monthly historical data aligned to a monthly time grid.
	Annual data are copied for each month, and sub-monthly data are averaged on monthly level. 
	These are not necessarily representative of aggregate values, but in DySE they all normalize
	to [0,1], so it is ultimately irrelevant whether we are averaging or summing up to aggregate.
	"""
	df = pd.DataFrame(columns=sorted(indicator_data.keys()))
	t_min = np.nanmin([np.nanmin(v['timestamps']) for k,v in indicator_data.items()])
	t_max = np.nanmax([np.nanmax(v['timestamps']) for k,v in indicator_data.items()])
	T_min = pd.to_datetime(pd.Timestamp(t_min, unit='ms').date())
	T_max = pd.to_datetime(pd.Timestamp(t_max, unit='ms').date())
	JH = pd.date_range(start=T_min, end=T_max, freq='MS', closed=None)
	df['Date'] = JH
	for k,v in indicator_data.items():
		resolution = v['resolution']
		agg_dict = {tjh:[] for tjh in JH}
		for t, x in zip(pd.to_datetime(v['timestamps'], unit='ms'), v['values']):
			if resolution=='month':
				idx = 12*(t.year-T_min.year)+(t.month-T_min.month)
				if idx>=0 and idx<len(JH):
					agg_dict[JH[idx]].append(x)
			elif resolution=='year' and t.year>=T_min.year and t.year<=T_max.year:
				idx = 12*(t.year-T_min.year) - T_min.month + 1
				for j in range(12):
					if idx+j>=0 and idx+j<len(JH):
						agg_dict[JH[idx+j]].append(x)
				
		df[k] = df.apply(lambda x: np.nanmean(agg_dict[x['Date']]), axis=1)
	return df    

def fit_weights(elements, edges, df_data, steps_per_unit:int, method:str='linear', 
				delay:dict=None, fixed_edges:dict=dict(), min_weight:float=0.01):
	"""This function returns dictionary edges_dict with fitted weights where possible,
	and input weights otherwise.

	TODO: Use fixed_edges dictionary to encode edges with fixed weights. Format:
	{(source_node, target_node): {'source':source_node, 'target':target_node,
								  'level-weight':level_weight,
								  'trend-weight':trend_weight,
								  'polarity': edge_polarity}}"""


	# Ascertain weights fitting method is supported
	assert method.lower() in ['nnls','linear','lasso'], "Method {0} unknown! Select a method from \{nnls, linear, lasso\}".format(method)
	method = method.lower()

	# Initialize edges_dict dictionary to store the inferred weights and return them
	edges_dict = dict()

	# If no delays on interactions are provided, initialize with default delay of 1 time step for all
	# NOTE: these delays are target-oriented, i.e. they only tell you how much later will the target
	# be updated
	if delay is None:
		delay = {el:1 for el in elements}

	# Iterate over target elements
	for target in elements:
		sources = [e['source'] for e in edges.values() if e['target']==target]
		if len(sources)==0:
			continue
		polarities = np.array([e['polarity'] for e in edges.values() if e['target']==target]*2).reshape(-1,1)
		try:
			# Attempt at extracting overlapping historical level and trend values to infer weights
			# This first phase is only inferring regulation types

			# Build level-based features
			X_level = np.concatenate([discretize(df_data[:-delay[target]][s].values.reshape(-1,1)) for s in sources],axis=1)
			X_level_reg = (X_level[:-1] + X_level[1:]) * steps_per_unit / 2.

			# Build trend-based features
			X_trend = X_level[1:-delay[target],:]-X_level[:-delay[target]-1,:]
			X_trend = np.concatenate([X_trend[0,:].reshape(1,-1), X_trend], axis=0)

			# Join level-based and trend-based features
			X = np.concatenate([X_level_reg, X_trend], axis=1)

			# Apply polarities
			X = X * np.dot(np.ones((X.shape[0],1)),polarities.T)

			# Build discretized output values
			y = discretize(df_data[delay[target]:][target].values).reshape(-1,1)
			y_update = y[1:,:]-y[:-1,:]

			# Join X and y
			data4fit = np.concatenate([y_update.reshape(-1,1),X], axis=1)

			# Remove NaNs (rows with missing values)
			data4fit = cut_NaNs(data4fit, axis=0)

			# Calculate correlations to infer regulation types
			corr = np.corrcoef(data4fit.T)[0,1:]
			
			# Split the outputs from the features
			y_update_cut, X_cut = data4fit[:,0].reshape(-1), data4fit[:,1:]
			inferred = True
		except:
			inferred = False
			corr = np.zeros(2*len(sources))

		# Reiterate fitting for inferred regulations types
		if inferred:
			regulation_types = dict()
			selected_idx = []
			for i,s in enumerate(sources):
				if corr[i] > corr[i+len(sources)] and corr[i]>0:
					regulation_types[(s,target)] = 'level-based'
					selected_idx.append(i)
				else:
					regulation_types[(s,target)] = 'trend-based'
					selected_idx.append(i+len(sources))

			X2 = np.concatenate([y_update, X[:,selected_idx]], axis=1)
			X2 = cut_NaNs(X2, axis=0)
			y_update_cut, X2 = X2[:,0].reshape(-1), X2[:,1:]
			try:
				if method=='nnls':
					fit_selected_values = nnls(A=X2, b=y_update_cut)[0]
				elif method=='linear':
					LR = LinearRegression()
					fit_selected_values = LR.fit(X=X2, y=y_update_cut).coef_
				elif method=='lasso':
					LASSO = Lasso()
					fit_selected_values = LASSO.fit(X=X2[1:], y=y_update_cut).coef_
				else:
					raise ValueError("Method {0} unknown! Select a method from \{nnls, linear, lasso\}".format(method))
				fit_selected_values = np.where(fit_selected_values>0, fit_selected_values, min_weight)
				inferred = True
			except:
				fit_selected_values = [edges[(s,target)]['level-weight'] for s in sources]\
							+[edges[(s,target)]['trend-weight'] for s in sources]
				inferred = False

		for i,s in enumerate(sources):
			if inferred:
				if regulation_types[(s,target)]=='level-based':
					w_level = max([int(fit_selected_values[i]*100000.)/100000, min_weight])
					w_trend = 0
				elif regulation_types[(s,target)]=='trend-based':
					w_level = 0
					w_trend = max([int(fit_selected_values[i]*100000.)/100000, min_weight])
				else:
					w_level = 0
					w_trend = 0.5
				edges_dict[(s,target)] = {'source':s, 'target':target,
										 'level-weight':w_level,
										 'trend-weight':w_trend,
										 'polarity':polarities[i][0]}
			else:
				if edges[(s,target)]['level-weight']<min_weight and \
					edges[(s,target)]['trend-weight']<min_weight:
					edges[(s,target)]['trend-weight'] = min_weight
				edges_dict[(s,target)] = {'source':s, 'target':target,
								 'level-weight':edges[(s,target)]['level-weight'],
								 'trend-weight':edges[(s,target)]['trend-weight'],
								 'polarity':edges[(s,target)]['polarity']}

	return edges_dict

def name_variable(name:str):
	"""This function edits variable name to make it conform to DySE specifications:
	- it replaces '/' with '___'
	- it replaces ' ' with '__'
	- it adds '_' in front of the variable name if it starts with a digit"""
	new_name = name.replace('/','___').replace(' ','__')
	if re.match(r'^\d', new_name):
		new_name = '_'+str(new_name)
	return new_name

def create_model(model_json:str, model_name:str='model.xslx', infer_weights:bool=True, default_level:int=15,
				 default_trend_weight:float=0.5, default_level_weight:float=0.0):
	"""This function creates DySE model spreadsheet from the model_json file."""
	json_model = json.load(open(model_json,'r'))
	df_model = pd.DataFrame(columns=['Variable',
									 '#',
									 'Element Name',
									 'Element IDs',
									 'Element ID',
									 'Element Type',
									 'Agent',
									 'Patient',
									 'Value Judgment',
									 'Specificity',
									 'Location',
									 'Time Scale / Frequency',
									 'Value: Activity / Amount ',
									 'Element NOTES',
									 'Positive',
									 'Negative',
									 'Influence Set NOTES',
									 'Levels',
									 'Spontaneous Behavior',
									 'Balancing Behavior',
									 'Update Group',
									 'Update Rate',
									 'Update Rank',
									 'Delay',
									 'Mechanism',
									 'Weight',
									 'Regulator Level',
									 'Evidence',
									 'Initial 0'])

	if 'weights' in json_model['edges']:
		edges = {(x['source'],x['target']):{'source':x['source'], 'target':x['target'],
									   'level-weight':x['weights'][0], 'trend-weight':x['weights'][1],
									   'polarity':1} for x in json_model['edges']}
	else:
		edges = {(x['source'],x['target']):{'source':x['source'], 'target':x['target'],
									   'level-weight':default_level_weight, 
									   'trend-weight':default_trend_weight,
									   'polarity':1} for x in json_model['edges']}
	
	if 'conceptIndicators' in json_model:
		elements = sorted(json_model['conceptIndicators'].keys())
		lvl_dict = {k:v['numLevels'] for k,v in json_model['conceptIndicators'].items()}
		for st in json_model['statements']:
			edges[(st['subj']['concept'],st['obj']['concept'])]['polarity'] = \
															st['subj']['polarity']*st['obj']['polarity']
	elif 'nodes' in json_model:
		elements = sorted([node['concept'] for node in json_model['nodes']])
		lvl_dict = {node['concept']:node['numLevels'] for node in json_model['nodes']}
		for edge in json_model['edges']:
			edges[(edge['source'], edge['target'])]['polarity'] = edge['polarity']

	else:
		raise ValueError("Neither 'conceptIndicators', nor 'nodes' found in the model JSON.")
	df_model['Element IDs'] = elements
	df_model['Element ID'] = elements
	df_model['#'] = range(1, len(elements)+1)
	df_model['Balancing Behavior'] = ['None']*len(elements)
	df_model['Spontaneous Behavior'] = ['None']*len(elements)
	df_model['Levels'] = df_model.apply(lambda x: lvl_dict[x['Element IDs']], axis=1)
	df_model['Initial 0'] = default_level
	df_model['Element Name'] = df_model.apply(lambda x: name_variable(x['Element IDs']), axis=1)
	df_model['Variable'] = df_model['Element Name']
	df_model.to_excel(model_name,index=False)
		
	update_model(original_model=model_name, updated_model=model_name, edges=edges)

	# Infer the weights
	if infer_weights:
		weights_inference(json_model=json_model, elements=elements, edges=edges, model_name=model_name)

def weights_inference(json_model:dict, elements:list, edges:dict, model_name:str, cycle_length:float=0):
	"""This function infers weights from the historical data (provided in json_model JSON file).

	Arguments:
	- json_model [dict]: a JSON file with CauseMos model information (including historical data)
	- elements [list]: a list of elements for whose incoming edges to perform weights inference on
	- edges [dict]: a dictionary with all the edges, where the key is a tuple (source, target),
					and the value is a dictionary storing source, target, 'level-weight', 
					'trend-weight', and 'polarity' (1 or -1)
	- model_name [str]: the name of the model Excel spreadsheet
	- cycle_length [float]: number of time steps per the lowest time unit in historical data
							(default: 0, to automatically calculate as the number of updateable
							elements)"""

	df_model = pd.read_excel(model_name).fillna('')
	if cycle_length==0:
		cycle_length = df_model.shape[0]-len(find_heads(model_name))

	# Build a DataFrame with indicator data
	if 'conceptIndicators' in json_model:
		indicator_dict = {k:dict() for k in json_model['conceptIndicators']}
		for k,v in json_model['conceptIndicators'].items():
			indicator_dict[k]['timestamps'] = [y['timestamp'] for y in v['values']]
			indicator_dict[k]['values'] = [y['value'] for y in v['values']]
			indicator_dict[k]['resolution'] = v['resolution']
	elif 'nodes' in json_model:
		indicator_dict = {node['concept']:dict() for node in json_model['nodes']}
		for node in json_model['nodes']:
			indicator_dict[node['concept']]['timestamps'] = [y['timestamp'] for y in node['values']]
			indicator_dict[node['concept']]['values'] = [y['value'] for y in node['values']]
			indicator_dict[node['concept']]['resolution'] = node['resolution']
	else:
		raise ValueError("Neither 'conceptIndicators', nor 'nodes' found in the model JSON.")

	df_data = build_df_data(indicator_data=indicator_dict)
	
	# Infer weights
	edges = fit_weights(elements=elements, edges=edges, steps_per_unit=cycle_length, 
						df_data=df_data, method='nnls')
	
	update_model(original_model=model_name, updated_model=model_name, edges=edges)

class CauseMosIndicators:
	"""This class handles CauseMos indicator data loading, processing, management, and visualization."""
	def __init__(self, model_ID=None):
		self.model_ID = model_ID
		self.historical_data = None
		self.time_series = None
		self.interpolated_time_series = None
		self.history_timelines = None
		self.frequencies = None
		self.indicator_names = None
		self.indicators = None
		self.baseline_projections = None
		self.grid_data = None
		self.joint_projection_timeline = []
		self.stat_forecast_slice = None

	def find_freq(self, times, values):
		"""This function determines temporal resolution of a time series.
		
		Inputs:
		- times [iterable]: sorted list of timestamps
		- values [iterable]: values at the times' timestamps
		
		Output:
		- freq [str]: temporal resolution to be chosen from 
					  \{'second','minute','hour','day','week','month', or 'year'\}"""

		assert len(times)==len(values), "Lengths of times and values need to be equal"\
										"while here they are {0} and {1}, respectfully"\
										.format(len(times),len(values))
		t_array = np.array(sorted([t for t,v in zip(times,values) if not np.isnan(v)]))  
		min_dt = np.min(t_array[1:]-t_array[:-1])
		if min_dt>=pd.Timedelta('365D'):
			return 'year'
		elif min_dt>=pd.Timedelta('4W'):
			return 'month'
		elif min_dt>=pd.Timedelta('1W'):
			return 'week'
		elif min_dt>=pd.Timedelta('1D'):
			return 'day'
		elif min_dt>=pd.Timedelta('1H'):
			return 'hour'
		elif min_dt>=pd.Timedelta('1M'):
			return 'minute'
		elif min_dt>=pd.Timedelta('1S'):
			return 'second'
		elif min_dt>=pd.Timedelta('1L'):
			return 'millisecond'
		elif min_dt>=pd.Timedelta('1U'):
			return 'microsecond'
		elif min_dt>=pd.Timedelta('1N'):
			return 'nanosecond'
		else:
			return '0'
		
	def load_data(self, data_json:str, request_json:str, history_start_date:pd.Timestamp=None, history_end_date:pd.Timestamp=None):
		"""This function loads historical data json file and returns data dictionary."""
		with open(data_json,'r') as djson, open(request_json,'r') as rjson:
			model_dump = json.load(djson)
			req_json = json.load(rjson)
			hist_data = dict()
			hist_timeline = dict()
			time_series = dict()
			ts_with_nans = dict()
			freq = dict()
			freq_unit = {'year':'YS', 'month':'MS', 'week':'W', 'day':'D', 'hour':'H', 'minute':'T',
						'second':'S', 'millisecond':'L', 'microsecond':'U', 'nanosecond':'N', '0':None}
			final_freq = 'nanosecond'
			final_timedelta = pd.Timedelta('1N')

			if not 'conceptIndicators' in model_dump and not 'nodes' in model_dump:
				raise ValueError("Neither 'conceptIndicators', nor 'nodes' found in the model JSON.")

			if 'experimentParam' in req_json:
				expParam = 'experimentParam'
			elif 'experimentParams' in req_json:
				expParam = 'experimentParams'
			else:
				raise ValueError('experimentParams missing from the experiment request json')
			projection_start_date = pd.Timestamp(req_json[expParam]['startTime'], unit='ms')
			projection_end_date = pd.Timestamp(req_json[expParam]['endTime'], unit='ms')
			
			if 'conceptIndicators' in model_dump:
				for k,v in model_dump['conceptIndicators'].items():
					if len(v['values'])==0:
						hist_data[k] = {'values':None}
						freq[k] = 'month'
						hist_timeline[k] = None 
						ts_with_nans[k] = None 
						time_series[k] = None 
					
					else:
						vals, times = zip(*sorted(map(lambda x: (x['value'],x['timestamp']),v['values']),key=lambda x: x[1]))
						vals = np.array(vals)
						times = np.array([pd.Timestamp(x,unit='ms') for x in times])
						hist_data[k] = {'timestamps':times, 'values':vals}
						freq[k] = self.find_freq(times, vals)
						if freq[k]=='0':
							raise ValueError('Invalid historical data frequency!')
						if freq[k]=='year':
							final_freq = 'year'
							final_timedelta = pd.Timedelta('1Y')
						elif freq[k]=='month' and not final_freq=='year' and not final_freq=='month':
							final_freq = 'month'
							final_timedelta = pd.Timedelta('28D')
						elif pd.Timedelta('1'+freq_unit[freq[k]]) > final_timedelta:
							final_freq = freq[k]
							final_timedelta = pd.Timedelta('1'+freq_unit[freq[k]])
						
						if history_start_date==None:
							start_date = np.nanmin(times)
						else:
							start_date = history_start_date

						closed_arg = None
						if history_end_date==None:
							end_date = np.nanmin([np.nanmax(times),projection_start_date])
							if end_date == projection_start_date:
								closed_arg = 'left'
						else:
							end_date = history_end_date

						hist_timeline[k] = pd.date_range(start=start_date, end=end_date, freq=freq_unit[freq[k]], closed=closed_arg)
						val_binned = {t:[] for t in hist_timeline[k]}
						for t,v in zip(times,vals):
							if t>=projection_start_date:
								continue
							try:
								idx = np.where(hist_timeline[k]>=t)[0][0]
								val_binned[hist_timeline[k][idx]].append(v)
							except:
								pass

						processed_data = np.array([np.nan for ht in hist_timeline[k]])
						for i,ht in enumerate(hist_timeline[k]):
							if len(val_binned[ht])!=0:
								processed_data[i] = np.nanmean(val_binned[ht])
						time_series[k] = fill_nan(processed_data)
						ts_with_nans[k] = processed_data
					name_dict = {k:model_dump['conceptIndicators'][k]['name'] \
								if 'name' in model_dump['conceptIndicators'][k] \
								else None for k in model_dump['conceptIndicators']}

			elif 'nodes' in model_dump:
				for node in model_dump['nodes']:
					if len(node['values'])==0:
						hist_data[node['concept']] = {'values':None}
						freq[node['concept']] = 'month'
						hist_timeline[node['concept']] = None 
						ts_with_nans[node['concept']] = None 
						time_series[node['concept']] = None 
					
					else:
						vals, times = zip(*sorted(map(lambda x: (x['value'],x['timestamp']), node['values']),
												  key=lambda x: x[1]))
						vals = np.array(vals)
						times = np.array([pd.Timestamp(x,unit='ms') for x in times])
						hist_data[node['concept']] = {'timestamps':times, 'values':vals}
						freq[node['concept']] = self.find_freq(times, vals)
						if freq[node['concept']]=='0':
							raise ValueError('Invalid historical data frequency!')
						if freq[node['concept']]=='year':
							final_freq = 'year'
							final_timedelta = pd.Timedelta('1Y')
						elif freq[node['concept']]=='month' and not final_freq=='year' and not final_freq=='month':
							final_freq = 'month'
							final_timedelta = pd.Timedelta('28D')
						elif pd.Timedelta('1'+freq_unit[freq[node['concept']]]) > final_timedelta:
							final_freq = freq[node['concept']]
							final_timedelta = pd.Timedelta('1'+freq_unit[freq[node['concept']]])
						
						if history_start_date==None:
							start_date = np.nanmin(times)
						else:
							start_date = history_start_date

						closed_arg = None
						if history_end_date==None:
							end_date = np.nanmin([np.nanmax(times),projection_start_date])
							if end_date == projection_start_date:
								closed_arg = 'left'
						else:
							end_date = history_end_date

						hist_timeline[node['concept']] = pd.date_range(start=start_date, end=end_date, 
														freq=freq_unit[freq[node['concept']]], closed=closed_arg)
						val_binned = {t:[] for t in hist_timeline[node['concept']]}
						for t,v in zip(times,vals):
							if t>=projection_start_date:
								continue
							try:
								idx = np.where(hist_timeline[node['concept']]>=t)[0][0]
								val_binned[hist_timeline[node['concept']][idx]].append(v)
							except:
								pass

						processed_data = np.array([np.nan for ht in hist_timeline[node['concept']]])
						for i,ht in enumerate(hist_timeline[node['concept']]):
							if len(val_binned[ht])!=0:
								processed_data[i] = np.nanmean(val_binned[ht])
						time_series[node['concept']] = fill_nan(processed_data)
						ts_with_nans[node['concept']] = processed_data
				name_dict = {node['concept']:node['indicator'] \
							if 'indicator' in node \
							else None for node in model_dump['nodes']}
			else:
				raise ValueError("Neither 'conceptIndicators', nor 'nodes' found in the model JSON.")

			self.historical_data = hist_data
			self.time_series = ts_with_nans
			self.interpolated_time_series = time_series
			self.history_timelines = hist_timeline
			self.frequencies = freq
			self.indicator_names = name_dict
			if 'conceptIndicators' in model_dump:
				self.indicators = model_dump['conceptIndicators']
			elif 'nodes' in model_dump:
				self.indicators = {node['concept']:{'name':node['indicator'],
													'values':node['values'],
													'numLevels':node['numLevels'],
													'resolution':node['resolution'],
													'period':node['period'],
													'minValue':node['minValue'],
													'maxValue':node['maxValue']} \
										for node in model_dump['nodes'] \
										if 'concept' in node and \
										'indicator' in node and \
										'values' in node and \
										'numLevels' in node and  \
										'resolution' in node and \
										'period' in node and \
										'minValue' in node and \
										'maxValue' in node \
										}
			else:
				raise ValueError("Neither 'conceptIndicators', nor 'nodes' found in the model JSON.")

			self.projection_start = projection_start_date
			self.projection_end = projection_end_date

	def slice_by_history(self, element, history):
		"""Returns historical data for an element sliced by the historical range ('history')"""
		H_dict = defaultdict(lambda: np.nan, zip(self.historical_data[element]['timestamps'],
												self.historical_data[element]['values']))
		return np.array([H_dict[t] for t in history])

	def joint_history(self, elements:list, combination_method:str='intersection'):
		"""This method yields a joint timeline for all the listed elements. Combination
		method determines whether to return the overlapping interval ('intersection')
		or does it return the smallest encompassing time interval ('union')."""
		starts, ends = [], []
		for k, v in self.historical_data.items():
			if k in elements and not v['values'] is None and not v['values']==[]:
				min_v, max_v = np.nanmin(v['timestamps']), np.nanmax(v['timestamps'])
				starts.append(min_v)
				ends.append(max_v)

		# determine the frequency for the joint timeline
		# TODO: Allow higher frequencies (e.g. minutes - it would need to work with something
		# different than pandas.date_range())
		if 'day' in self.frequencies.values():
			freq = 'D'
		elif 'week' in self.frequencies.values():
			freq = 'WS'
		elif 'month' in self.frequencies.values():
			freq = 'MS'
		else:
			freq = 'AS'

		# Return the constructed date range over the combined time interval
		if len(starts)>0:
			closed_arg = None
			if combination_method=='intersection':
				hist_start, hist_end = np.nanmax(starts), np.nanmin(ends)   
			elif combination_method=='union':
				hist_start, hist_end = np.nanmin(starts), np.nanmax(ends)
			# if hist_end>=self.projection_start:
			# 	closed_arg = 'left'
			# 	hist_end = self.projection_start
			return pd.date_range(hist_start, hist_end, freq=freq, closed=closed_arg)
		return None

	def baseline_projection(self, data:dict, time_series:dict, history_start_date, history_end_date, 
							history_timelines, projection_start_date, projection_end_date,
							freq:list, out_path:str='.', visualize:bool=False, verbose:bool=False):
		"""This function produces baseline projections for the elements given their historical data.
		
		Inputs:
		- data [dict]: dictionary with 'conceptIndicators' or 'nodes' data
		- time series [dict]: extracted historical time series for each element
		- history_start_date [pandas.Timestamp]: historical data start date
		- history_end_date [pandas.Timestamp]: historical data end date
		- history_timelines [dict]: dictionary containing historical timelines for each element
		- projection_start_date [pandas.Timestamp]: start date
		- projection_end_date [pandas.Timestamp]: end date
		- freq [list]: list of strings representing temporal resolution of data for each variable

		
		Outputs:
		- self.baseline_projections [dict]: dictionary with the same structure as data dictionary
		- self.joint_projection_timeline [pd.DateIndex]: date range for the joint projection timeline"""
		
		# Process history/projection start/end date
		if type(history_start_date)==str:
			if re.match(r'\d{4}\-\d{2}\-\d{2}',history_start_date):
				history_start_date = pd.to_datetime(pd.Timestamp(history_start_date),unit='ms')
			else:
				history_start_date = pd.to_datetime(history_start_date,unit='ms')
		if type(history_end_date)==str:
			if re.match(r'\d{4}\-\d{2}\-\d{2}',history_end_date):
				history_end_date = pd.to_datetime(pd.Timestamp(history_end_date),unit='ms')
			else:
				history_end_date = pd.to_datetime(history_end_date,unit='ms')
		if type(projection_start_date)==str:
			if re.match(r'\d{4}\-\d{2}\-\d{2}',projection_start_date):
				projection_start_date = pd.to_datetime(pd.Timestamp(projection_start_date),unit='ms')
			else:
				projection_start_date = pd.to_datetime(projection_start_date,unit='ms')
		if type(projection_end_date)==str:
			if re.match(r'\d{4}\-\d{2}\-\d{2}',projection_end_date):
				projection_end_date = pd.to_datetime(pd.Timestamp(projection_end_date),unit='ms')
			else:
				projection_end_date = pd.to_datetime(projection_end_date,unit='ms')

		# Fix seasonality parameters
		seasonality_dict = {'MS':12, 'AS':None, 'D':7, 'W':52, 'H':24, 'M':None, 'S': None,
						   'L': None, 'U':None, 'N':None}
		seasonality_type = 'add'

		# Fix frequency arguments
		freq_dict = {'year':'AS', 'month':'MS', 'week':'W', 'day':'D', 'hour':'H', 'minute':'M',
				   'second':'S', 'millisecond':'L', 'microsecond':'U', 'nanosecond':'N'}
		freq_list = ['AS','MS','W','D','H','M','S','L','U','N']
		freqs_type = {v:i for i,v in enumerate(freq_list)}
		freq_idx = np.max([freqs_type[freq_dict[v]] for k,v in freq.items()])
		
		# Constructing timelines
		joint_proj_timeline = pd.date_range(start=projection_start_date,
											end=projection_end_date,
											freq=freq_list[freq_idx])

		# Projections
		projections = dict()
		for k,v in time_series.items():
			if v is None:
				projections_timeline = pd.date_range(start=projection_start_date,
													 end=projection_end_date,
												 	 freq='MS')
				num_points = len(projections_timeline)
				df_projections = pd.DataFrame()
				df_projections['mean'] = np.ones(num_points)
				df_projections['pi_lower'] = np.zeros(num_points)
				df_projections['pi_upper'] = np.ones(num_points)*2
				df_projections.index = projections_timeline
				projections[k] = df_projections
				
			else:
				projections_timeline = pd.date_range(start=projection_start_date,
													 end=projection_end_date,
													 freq=freq_dict[freq[k]])
				num_points = len(projections_timeline)
				if verbose:
					print("Building baseline projections for:",k.split('/')[-1])
				
				not_nans = np.where(~np.isnan(v))[0]
				if len(not_nans)==0:
					projections[k] = np.ones(len(v)+num_points)*np.nan
				else:    
					n_start = not_nans[0]
					n_end = not_nans[-1]
					seasonal_period = seasonality_dict[freq_list[freq_idx]]
					if seasonal_period is None:
						seasonality_type = None
					elif seasonal_period>(n_end-n_start+1)/2:
						seasonal_period = None
						seasonality_type = None
					else:
						seasonality_type = 'add'
						
					# Build a DataFrame with aligned data
					df_data = pd.DataFrame()
					df_data['value'] = np.array(fill_nan(v))
					df_data.index = history_timelines[k]
					
					if sum(~pd.isna(v))>10:
						init_method = 'heuristic'
						ETS_fit = sm.tsa.ETSModel(df_data[n_start:n_end+1]['value'],trend='add',damped_trend=True,
									seasonal=seasonality_type,seasonal_periods=seasonal_period,
									initialization_method=init_method).fit()
					else:
						init_method = 'known'
						ETS_fit = sm.tsa.ETSModel(df_data[n_start:n_end+1]['value'],trend='add',damped_trend=True,
									seasonal=seasonality_type,seasonal_periods=seasonal_period,
									initialization_method=init_method, initial_level=df_data.iloc[n_start,0],
											 initial_trend=0, initial_seasonal=seasonal_period).fit()
						
					projections[k] = ETS_fit.get_prediction(end=projection_end_date).summary_frame(alpha=0.05)
					if visualize:
						self.visualize_projections(projection=projections[k], hist_data=df_data, k=k, name=data[k]['name'],
										  hist_start=history_start_date, hist_end=history_end_date,
										  proj_start=projection_start_date, proj_end=projection_end_date,
										  out_path='dev_demo', filetype='jpg', dpi=150) 
		
		self.baseline_projections = projections 
		self.stat_forecast_slice = {k:v[joint_proj_timeline[0]:joint_proj_timeline[-1]] \
									for k,v in projections.items()}
		self.joint_projection_timeline = joint_proj_timeline  

	def visualize_projections(self, projection, hist_data:pd.DataFrame, k:str, name:str, 
							  hist_start:str, hist_end:str, proj_start:str, proj_end:str,
							  out_path:str='.', filetype:str='png',
							  dpi:int=150):
		
		# Clear canvas
		plt.clf()
		plt.cla()

		# historical time grid construction
		data_end = hist_data.index[np.where(~pd.isna(hist_data['value']))[0][-1]]

		# plot historical data
		plt.plot(hist_data,'k-o',label='Historical data')

		# plot predictions in historical range
		plt.plot(projection[hist_start:hist_end]['mean'],c='tab:grey',linestyle='--',label='Predicted historical data')
		plt.fill_between(projection[hist_start:hist_end].index,
						projection[hist_start:hist_end]['pi_lower'],
						projection[hist_start:hist_end]['pi_upper'],
						alpha=0.3, color='tab:grey')

		# plot predictions in forecasting range
		plt.plot(projection[proj_start:proj_end]['mean'],c='tab:blue',linestyle='--',label='Projections')
		plt.fill_between(projection[proj_start:proj_end].index,
						projection[proj_start:proj_end]['pi_lower'],
						projection[proj_start:proj_end]['pi_upper'], alpha=0.3, color='tab:blue')

		# label axes
		plt.xlabel('Time')
		plt.ylabel(name)

		# set x-axis bounds
		plt.xlim(hist_start,proj_end)

		# set legend
		plt.legend(bbox_to_anchor=(1.5, 1), loc='upper right')

		# set title
		plt.title(k.split('/')[-1])

		# save the plot to a file
		if out_path!='.' and not os.path.exists(out_path):
			os.mkdir(out_path)
		plt.savefig(os.path.join(out_path,k.split('/')[-1]+"."+filetype), bbox_inches='tight', dpi=dpi)

	def visualize_historical_data(self, hist_data:dict, element:str, name_dict:dict, 
									hist_start, hist_end, proj_start, proj_end, savefile=None):
		"""Visualization of historical data"""
		fig, ax = plt.subplots()
		ax.plot(hist_data[element]['timestamps'],hist_data[element]['values'],'k-o',label='Historical data')
		ax.set_xlim(pd.Timestamp(hist_start), pd.Timestamp(proj_end)+pd.Timedelta('1D'))
		y_min = np.nanmin(hist_data[element]['values'])
		y_max = np.nanmax(hist_data[element]['values'])
		y_bottom = 2 * y_min - y_max
		y_top = 2 * y_max - y_min
		ax.set_ylim(y_bottom, y_top)
		ax.axvspan(hist_start, hist_end,color='grey',alpha=0.1)
		ax.set_xlabel('Time')
		ax.set_ylabel(name_dict[element])
		ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0))
		ax.set_title(element.split('/')[-1].replace('_',' '))
		if savefile:
			plt.savefig(savefile, bbox_inches='tight', dpi=150)
		return ax

def main():
	return

if __name__ == "__main__":
	main()
