from Simulation.Simulator_Python.simulator import Simulator, Element
import pandas as pd
import sys
import logging
import numpy as np
from collections import defaultdict


# Extract the weights from the model spreadsheet, for weighted sum model only
# Useful when conducting Function-based discrete sensitivity
def element_regulator_weight_dict(element, model_excel):
    weight_dict = defaultdict(lambda:0.0)
    trend_weight_dict = defaultdict(lambda:0.0)
    data = pd.read_excel(model_excel)
    data = data.where(pd.notnull(data), '')
    pos_expression = data.set_index('Element Name')['Positive'].to_dict()[element]
    neg_expression = data.set_index('Element Name')['Negative'].to_dict()[element]
    expression = str(pos_expression) + '+' + str(neg_expression)
    for reg_exp in [x for x in expression.split('+') if x != '']:
        if '*' in reg_exp:
            if '&' in reg_exp:
                reg = reg_exp.split('*')[1]
                lw = float(reg_exp.split('*')[0].split('&')[1])
                if weight_dict[reg] == 0.0:
                    weight_dict[reg] = weight_dict[reg] + (lw>0)*lw
                else:
                    weight_dict[reg] = weight_dict[reg] - (lw>0)*lw
                tw = float(reg_exp.split('*')[0].split('&')[0])
                if trend_weight_dict[reg] == 0.0:
                    trend_weight_dict[reg] = trend_weight_dict[reg] + (tw>0)*tw
                else:
                    trend_weight_dict[reg] = trend_weight_dict[reg] - (tw>0)*tw
            else:
                weight_dict[reg_exp.split('*')[1]] = abs(float(reg_exp.split('*')[0]))
                trend_weight_dict[reg_exp.split('*')[1]] = 0.0
        else:
            weight_dict[reg_exp] = 1.0
            trend_weight_dict[reg_exp] = 0.0
    return weight_dict, trend_weight_dict

# Determine the boundary range of an element with large number of levels
# during which table-based sensitivity is applied
def element_value_near_boundary(total_level, threshold):
    if total_level <= threshold:
        return list(range(total_level))
    else:
        study_range = min(int(total_level/2.0), int(threshold/2.0))
        study_list = list(range(total_level)[:study_range]) + list(range(total_level)[-study_range:])
        return study_list

# Find the state vector in the boundary ranges
def element_boundary_vector(element, model, b_study_threshold):
    level_dict={}
    for key, ele in model.get_elements().items():
        level_dict[ele.get_name()] = ele.get_levels()
    zip = [element_value_near_boundary(level_dict[element.get_name_list()[i]], b_study_threshold) for i in range(len(element.get_name_list()))]
    import itertools
    vectors = list(itertools.product(*zip))
    return vectors

# In scenario-based sensitivity analysis, trace file is loaded here
# into a dataframe, which is useful for estimating the joint state distributions
def load_traj_into_df(trace_file):
    data = pd.read_csv(trace_file, sep=" ", header=None, dtype='unicode')
    headers = data.iloc[0]
    df = pd.DataFrame(data.values[1:], columns=headers)
    df.drop(['time', 'step'], axis=1, inplace=True)
    return df

# Return the distributions of all possible joint state for a given element list, estimated from trajectories
# Different modes of estimation is supported:
# mode='AVG' when averaging through runs and through time steps
# more modes will be added soon.
def joint_state_dis(ele_list, trace_file, mode='AVG'):
    df = load_traj_into_df(trace_file)
    df['*joint_state'] = df[ele_list].apply(lambda x: ' '.join(x), axis=1)
    if mode == 'AVG':
        df2 = df['*joint_state'].reset_index().groupby('*joint_state').count()
        df2['frq'] = df2['index']/len(df.index)
        join_state_dict = df2['frq'].to_dict()
        return join_state_dict

# Return the joint state probability of a given element list being certain state vector, under a certain mode
def get_state_dis(ele_list, trace_file, state_string):
    my_dict = joint_state_dis(ele_list, trace_file)
    if state_string in my_dict:
        return my_dict[state_string]
    else:
        return 0.0

# This function evaluate the element given its regulators' state vector
# Both the input vector and output value are defined on integer space
def evaluate_element_at_vector(int_vector, element, model):
    array_dict={}
    for key, ele in model.get_elements().items():
        array_dict[ele.get_name()] = np.linspace(0, 1, ele.get_levels())
    current_state = [array_dict[element.get_name_list()[i]][int_vector[i]] for i in range(len(int_vector))]
    eva_state = element.evaluate_state(current_state)
    eva_state_index = array_dict[element.get_name()].tolist().index(eva_state)
    return eva_state_index

# This function returns the immediate influence of the n-th regulator on the regulated element
# if the current state vector of all regulators is given.
# Different modes of calculating influences are supported here:
# Norm=True, when the regulated element's change is normalized to its maximum range, guaranteeing to be within [0,1]
# LoSe=True, when only small changes to the studied regulator is taken into considerations
def etei_at_vector(int_vector, element, to_index, model, Norm=True, LoSe=True):
    level_dict={}
    for key, ele in model.get_elements().items():
        level_dict[ele.get_name()] = ele.get_levels()
    vectors_to_check = []
    lista = [i for i in range(level_dict[element.get_name_list()[to_index]]) if i!= int_vector[to_index]]
    for possible_change in lista:
        if LoSe == True:
            if abs(possible_change - int_vector[to_index]) == 1:
                new_vector = int_vector.copy()
                new_vector[to_index] = possible_change
                vectors_to_check.append(new_vector)
        else:
            new_vector = int_vector.copy()
            new_vector[to_index] = possible_change
            vectors_to_check.append(new_vector)
    sum = 0.0
    current_value = evaluate_element_at_vector(int_vector, element, model)
    for i in range(len(vectors_to_check)):
        new_value = evaluate_element_at_vector(vectors_to_check[i], element, model)
        sum = sum + abs(current_value-new_value)
    sum = sum/len(vectors_to_check)
    if Norm==True:
        sum = sum/(level_dict[element.get_name()]-1)
    return sum


# Generate the raw result of discrete sensitivity analysis using hybrid-based method(default)
# or function-based method if passing method='FB' which is faster but less accurate
# When passing trace_file, dynamic(i.e., scenario-dependent) sensitivity analysis is triggered,
# otherwise, static sensitivity analysis is applied
def generate_discrete_sa(model_excel, result_name, method, status_report=False, Norm=True, LoSe=True, trace_file=None):

    model = Simulator(model_excel)

    # Build a global element level dictionary for all model elements, useful for later coding
    level_dict = {}
    for key, ele in model.get_elements().items():
        level_dict[ele.get_name()] = ele.get_levels()

    f_out = open(result_name,'w')

    # Analyze element one by one, for each element, calculate the immediate influences of its regulators
    if status_report:
        print("Analyzing Sensitivity:\n")
    size = len(model.get_elements())
    status = 0
    for key, element in model.get_elements().items():
        f_out.write('{} = ({})\n'.format(element.get_name(),', '.join(element.get_name_list())))
        logging.info('Analysis for element'.format(element.get_name()))
        status = status + 1.0
        if status_report:
            print("{:.3%}\n".format(status/size))

        # The calculations of immediate influences is state-wise,
        # therefore all regulator joint vectors are enumerated
        #num_vector = np.prod([level_dict[element.get_name_list()[i]] for i in range(len(element.get_name_list()))])
        influence_boundary = 0.0
        influence_linear = 0.0
        boundary_ratio = 0.0

        # In linear region of both regulator and regulated elements(i.e., in these intermediate levels),
        # immediate influence is almost identical to the associated weight, normalized by the number of regulated element's levels
        # this holds true no matter static or scenario-dependent sensitivity is applied
        for to_index in range(len(element.get_name_list())-1):
            f_out.write('d{} / d{}\n'.format(element.get_name(), element.get_name_list()[to_index]))
            if (method == 'FB' or method == 'HB'):
                weight_dict, trend_weight_dict = element_regulator_weight_dict(element.get_name(), model_excel)
                influence_linear = abs(float(weight_dict[element.get_name_list()[to_index]])) + abs(float(trend_weight_dict[element.get_name_list()[to_index]]))
                if Norm == True:
                    influence_linear = float(influence_linear) / (element.get_levels()-1.0)

            # In hybrid implementation, boundary region is also taken into account,
            # firstly, boundary is to be determined based on the number of regulated element's levels
            if (method == 'HB') and (len(element.get_name_list())-1 <= 6):
                if len(element.get_name_list())-1 <= 2:
                    b_study_threshold = 6.0
                elif len(element.get_name_list())-1 <= 3:
                    b_study_threshold = 4.0
                else:
                    b_study_threshold = 2.0
                b_vectors = element_boundary_vector(element, model, b_study_threshold)

                # Boudary ratio measures how much boundary region covers the whole state space,
                # in static analysis, it is equal to the number ratio of boundary vectors to all vectors
                num_vector = np.prod([level_dict[element.get_name_list()[i]] for i in range(len(element.get_name_list()))])
                boundary_ratio = float(len(b_vectors))/num_vector

                f_out.write(' '.join(element.get_name_list())+'  influ_or_not   value\n')

                # Dynamic sensitivity analysis is triggered if there is trace file
                if trace_file:
                    # return the dictionary of all joint state existing in a trace file with its distribution as key
                    state_dis = joint_state_dis(element.get_name_list(), trace_file)

                    # For each state to be studied, if it falls into the boundary region,
                    # return the immediate influence etei_at_vector() at this state,
                    # then multiply it with associated state distribution, sum up to obtain influ_sum, pass it to influence_boundary
                    influ_sum = 0.0
                    bounary_sum_dis = 0.0
                    for vector_str in state_dis.keys():
                        int_vector = [int(x) for x in vector_str.split(' ')]
                        if tuple(int_vector) in b_vectors:
                            sensi = etei_at_vector(int_vector, element, to_index, model, Norm, LoSe)
                            influ_sum = influ_sum + sensi*state_dis[vector_str]

                            # Also, record the summation of state distribution of state in the boundary region
                            bounary_sum_dis = bounary_sum_dis + state_dis[vector_str]
                            if sensi*state_dis[vector_str] != 0.0:
                                output_str = ','.join([str(element) for element in int_vector])
                                # Output the states in boundary region and their associated immediate influence
                                f_out.write('{}  {}   {}\n'.format(output_str,str(int(1)),str(round(sensi, 5))))

                    # in dynamic analysis, boundary ratio is updated to the total distribution of boundary vectors within all vectors
                    boundary_ratio = bounary_sum_dis

                    influence_boundary = influ_sum

                # In static sensitivity analysis, all states in the boundary region is to be studied
                # for each state, return the immediate influence etei_at_vector() at this state,
                else:
                    influ_sum = 0.0
                    for int_vector in b_vectors:
                        int_vector = list(int_vector)
                        sensi = etei_at_vector(int_vector, element, to_index, model, Norm, LoSe)

                        # Equivalently, all vectors have the same uniform distribution
                        vector_distribution = 1.0/num_vector
                        influ_sum = influ_sum + sensi*vector_distribution
                        if sensi*vector_distribution != 0.0:
                            vector_str = ','.join([str(element) for element in int_vector])
                            # Output the states in boundary region and their associated immediate influence
                            f_out.write('{}  {}   {}\n'.format(vector_str,str(int(1)),str(round(sensi, 5))))

                    influence_boundary = influ_sum

            # The overall influence is summarized by linear and boundary regions
            overall_influence = influence_linear*(1-boundary_ratio) + influence_boundary
            overall_influence = min(1.0, overall_influence)
            f_out.write('-influence {}\n\n'.format(overall_influence))


def main():
    if len(sys.argv) < 3:
        raise ValueError("Missing arguments, at least model spreadsheet and output file name are required")
    model_excel = sys.argv[1]
    result_name = sys.argv[2]
    Norm = True
    LoSe = True
    method = 'HB'
    status_report = False
    trace_file = None
    try:
        trace_file = sys.argv[3]
    except:
        pass
    generate_discrete_sa(model_excel, result_name, method, status_report, Norm, LoSe, trace_file)

if __name__== "__main__" :
    main()
