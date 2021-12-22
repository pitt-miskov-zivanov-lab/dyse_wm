from Simulation.Simulator_Python.simulator import Simulator, Element
import pandas as pd
import sys
import logging


# Extract the weights from the model spreadsheet, for weighted sum model only
def element_regulator_weight_dict(element, model_excel):
    weight_dict={}
    data = pd.read_excel(model_excel)
    pos_expression = data.set_index('Element Name')['Positive'].to_dict()[element]
    neg_expression = data.set_index('Element Name')['Negative'].to_dict()[element]
    expression = str(pos_expression) + '+' + str(neg_expression)
    for reg_exp in [x for x in expression.split('+') if x != '']:
        if '*' in reg_exp:
            weight_dict[reg_exp.split('*')[1]] = reg_exp.split('*')[0]
        else:
            weight_dict[reg_exp] = 1.0
    return weight_dict

# Generate the raw result of static discrete sensitivity analysis using function-based method
def generate_discrete_static_sa(model_excel, result_name, Norm=True, method='FB'):
    if method == 'FB':
        model = Simulator(model_excel)
        level_dict={}
        for key, ele in model.get_elements().items():
            level_dict[ele.get_name()] = ele.get_levels()
        f_out = open(result_name,'w')
        for key, element in model.get_elements().items():
            f_out.write('{} = ({})\n'.format(element.get_name(),', '.join(element.get_name_list())))
            logging.info('Analysis for element'.format(element.get_name()))
            for to_index in range(len(element.get_name_list())-1):
                f_out.write('d{} / d{}\n'.format(element.get_name(), element.get_name_list()[to_index]))
                f_out.write(' '.join(element.get_name_list())+'\n')
                influence = element_regulator_weight_dict(element.get_name(),model_excel)[element.get_name_list()[to_index]]
                if Norm == True:
                    influence = float(influence)*(level_dict[element.get_name_list()[to_index]]-1.0) / (element.get_levels()-1.0)
                influence = min(1.0, influence)
                f_out.write('-weight {}\n\n'.format(influence))

def main():
    model_excel = sys.argv[1]
    result_name = sys.argv[2]
    Norm=True
    method='FB'
    generate_discrete_static_sa(model_excel, result_name, Norm, method)

if __name__== "__main__" :
    main()
