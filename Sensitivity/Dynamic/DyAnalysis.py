from Simulation.Simulator_Python.simulator import Simulator, Element
import numpy as np
import sys
import pandas as pd
import logging

# This function evaluate the element given its regulators' state vector
# Both the input vector and output value are defined on integer space
def evaluate_element_at_vector(int_vector, element, model):
    array_dict={}
    for key, ele in model.get_elements().items():
        array_dict[ele.get_name()] = np.linspace(0, 1, ele.get_levels())
    current_state = [array_dict[element.get_name_list()[i]][int_vector[i]] for i in range(len(int_vector)-1)]
    current_state.append(int_vector[-1])
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
    for possible_change in [i for i in range(level_dict[element.get_name_list()[to_index]]) if i!= int_vector[to_index]]:
        if LoSe == True:
            if abs(possible_change-int_vector[to_index]) == 1:
                new_vector = int_vector.copy()
                new_vector[to_index] = possible_change
                vectors_to_check.append(new_vector)
        else:
            new_vector = int_vector.copy()
            new_vector[to_index] = possible_change
            vectors_to_check.append(new_vector)
    sum = 0.0
    for i in range(len(vectors_to_check)):
        current_value = evaluate_element_at_vector(int_vector, element, model)
        new_value = evaluate_element_at_vector(vectors_to_check[i], element, model)
        sum = sum + abs(current_value-new_value)
    sum = sum/len(vectors_to_check)
    if Norm==True:
        sum = sum/(level_dict[element.get_name()]-1)
    return sum

# Load the trace file to be studied into a dataframe, which is useful to estimate
# the joint state distributions
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
def joint_state_dis(ele_list, df, mode):
    df['*joint_state'] = df[ele_list].apply(lambda x: ' '.join(x), axis=1)
    if mode == 'AVG':
        df2 = df['*joint_state'].reset_index().groupby('*joint_state').count()
        df2['frq'] = df2['index']/len(df.index)
        join_state_dict = df2['frq'].to_dict()
        return join_state_dict

# Return the joint state probability of a given element list being certain state vector, under a certain mode
def get_state_dis(ele_list, state_string, df, mode):
    if state_string in joint_state_dis(ele_list, df, mode):
        return joint_state_dis(ele_list, df, mode)[state_string]
    else:
        return 0.0

# Generate the raw result of dynamic discrete sensitivity analysis of the model on a trace DataFrame
def generate_discrete_dynamic_sa(model_excel, trace_file, result_name, Norm=True, LoSe=True, mode='AVG'):
    model = Simulator(model_excel)
    trace_df = load_traj_into_df(trace_file)
    f_out = open(result_name,'w')
    for key, element in model.get_elements().items():
        f_out.write('{} = ({})\n'.format(element.get_name(),', '.join(element.get_name_list())))
        logging.info('Analysis for element'.format(element.get_name()))
        element_acti = 0.0
        state_dis = joint_state_dis(element.get_name_list(), trace_df, mode)
        for to_index in range(len(element.get_name_list())):
            f_out.write('d{} / d{}\n'.format(element.get_name(), element.get_name_list()[to_index]))
            f_out.write(' '.join(element.get_name_list())+'\n')
            for vector_str in state_dis.keys():
                int_vector = [int(x) for x in vector_str.split(' ')]
                sensi = etei_at_vector(int_vector, element, to_index, model, Norm, LoSe)
                state_distri = state_dis[vector_str]
                element_acti = element_acti + sensi*state_distri
                if sensi != 0.0:
                    f_out.write('{}  {}   {}\n'.format(vector_str.replace(' ', ''),str(int(1)),str(round(sensi*state_distri, 10))))
        f_out.write('Overall Probability: {}\n\n'.format(str(round(element_acti,10))))

def main():
    model_excel = sys.argv[1]
    trace_file = sys.argv[2]
    result_name = sys.argv[3]
    Norm=True
    LoSe=True
    mode='AVG'
    generate_discrete_dynamic_sa(model_excel, trace_file, result_name, Norm, LoSe, mode)

if __name__== "__main__" :
    main()
