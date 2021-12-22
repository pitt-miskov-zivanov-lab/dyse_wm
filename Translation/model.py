import os
import sys
import pandas as pd
import argparse
import logging
import re
import json
import networkx as nx
from collections import defaultdict
import datetime
import numpy as np
import copy
from itertools import chain


# define regex for valid characters in variable names
_VALID_CHARS = r'a-zA-Z0-9\_'

# valid element types
_VALID_TYPES = [
    'protein', 'protein family', 'protein complex',
    'rna', 'mrna', 'gene', 'chemical', 'biological process'
    ]

_VAR_COL = 'Variable'
_IDX_COL = '#'

# TODO: make into a class definition for model object (reference simulator code)

def get_model_template(optional_columns=[], reading=pd.DataFrame()):

    model_columns = [
            'Variable',
            '#',
            'Element Name',
            'Element IDs',
            'Element Type',
            'Positive',
            'Negative',
            'Scenario']
    
    model_columns.extend(optional_columns)

    model_df = pd.DataFrame(columns=model_columns)

    if not reading.empty:
        # add ontology terms from reading
        elements = reading[
                ['Element Name','Element ID']
                ].rename(columns={
                        'Element Name' : 'Name',
                        'Element ID' : 'ID'
                        }).drop_duplicates()
        regulators = reading[
                ['Regulator Name','Regulator ID']
                ].rename(columns={
                        'Regulator Name' : 'Name',
                        'Regulator ID' : 'ID'
                        }).drop_duplicates()
        all_elements = elements.append(regulators).drop_duplicates()
        model_df['Variable'] = all_elements['Name']
        model_df['Element Name'] = all_elements['Name']
        model_df['Element IDs'] = all_elements['ID']
        model_df['#'] = model_df.reset_index().index
        model_df['Scenario'] = 1

    model_df.fillna('', inplace=True)
    
    return model_df


def get_model(model_file: str) -> pd.DataFrame:
    """Load model into a DataFrame and standardize column names
    """

    global _VALID_CHARS
    global _VAR_COL
    global _IDX_COL

    index_col_name = _IDX_COL
    var_col_name = _VAR_COL
    pos_reg_col_name = 'Positive'
    pos_list_col_name = 'Positive List'
    neg_reg_col_name = 'Negative'
    neg_list_col_name = 'Negative List'
    reg_list_col_name = 'Regulators'
    element_name_col_name = 'Element Name'
    ids_col_name = 'Element IDs'
    type_col_name = 'Element Type'

    # Load the input file containing elements and regulators
    model_sheets = pd.ExcelFile(model_file)
    # get the model from the first sheet, will check the other sheets for truth tables later
    model = model_sheets.parse(0,na_values='NaN',keep_default_na=False,index_col=None)

    # check model format
    if 'element attributes' in [x.lower() for x in model.columns]:
        # drop two header rows and set column names to third row
        model = model.rename(columns=model.iloc[1]).drop([0,1]).set_index(index_col_name)
    
    # get other sheets 
    # TODO: parse truth tables here? or just return other sheets separately?
    if len(model_sheets.sheet_names) > 1:
        df_other_sheets = {sheet : model_sheets.parse(sheet,na_values='NaN',keep_default_na=False) \
            for sheet in model_sheets.sheet_names[1:]}
    else:
        df_other_sheets = ''

    # format model columns 
    input_col_X = [
            x.strip() for x in model.columns 
            if ('variable' in x.lower())
            ]
    input_col_A = [
            x.strip() for x in model.columns 
            if ('positive' in x.lower())
            ]
    input_col_I = [
            x.strip() for x in model.columns 
            if ('negative' in x.lower())
            ]
    input_col_initial = [
            x.strip() for x in model.columns 
            if ('initial' in x.lower()
            or 'scenario' in x.lower())
            ]

    input_col_name = [
            x.strip() for x in model.columns 
            if ('element name' in x.lower())
            ]
    input_col_ids = [
            x.strip() for x in model.columns 
            if ('element ids' in x.lower())
            ]
    input_col_type = [
            x.strip() for x in model.columns 
            if ('element type' in x.lower())
            ]

    # check for all required columns or duplicate colummns
    if (len(input_col_X) == 0 
            or len(input_col_A) == 0
            or len(input_col_I) == 0 
            or len(input_col_initial) == 0
            ):
        raise ValueError(
                'Missing one or more required columns in input file: '
                'Variable, Positive, Negative, Scenario'
                )
    elif (len(input_col_X) > 1
            or len(input_col_A) > 1
            or len(input_col_I) > 1
            ):
        raise ValueError('Duplicate column of: Variable, Positive, Negative')

    if (len(input_col_name) == 0 
            or len(input_col_ids) == 0 
            or len(input_col_type) == 0
            ):
        raise ValueError(
                'Missing one or more required column names: '
                'Element Name, Element IDs, Element Type'
                )
    elif (len(input_col_name) > 1 
            or len(input_col_ids) > 1 
            or len(input_col_type) > 1
            ):
        raise ValueError(
                'Duplicate column of: '
                'Element Name, Element IDs, Element Type'
                )

    # TODO: check for other columns here as they are needed

    # processing
    # use # column or index to preserve order of elements in the model
    if index_col_name in model.columns:
        model.set_index(index_col_name,inplace=True)

    # remove rows with missing or marked indices
    model = drop_x_indices(model)

    model = model.reset_index()
    # standardize column names
    model = model.rename(
        index=str,
        columns={
            'index': index_col_name,
            input_col_X[0]: var_col_name,
            input_col_A[0]: pos_reg_col_name,
            input_col_I[0]: neg_reg_col_name,
            input_col_name[0]: element_name_col_name,
            input_col_ids[0]: ids_col_name,
            input_col_type[0]: type_col_name
        })

    # format invalid variable names
    model = format_variable_names(model)

    # standardize element types
    model['Element Type'] = model['Element Type'].apply(get_type)

    # set variable name as the index
    model.set_index(var_col_name,inplace=True)

    # check for empty indices
    if '' in model.index:
        raise ValueError('Missing variable names')
        # model = model.drop([''])

    # parse regulation functions into lists of regulators
    model[pos_list_col_name] = model[pos_reg_col_name].apply(
            lambda x: [y.strip() for y in re.findall('['+_VALID_CHARS+']+',x)]
            )
    model[neg_list_col_name] = model[neg_reg_col_name].apply(
            lambda x: [y.strip() for y in re.findall('['+_VALID_CHARS+']+',x)]
            )
    model[reg_list_col_name] = model.apply(
            lambda x: 
            set(list(x[pos_list_col_name]) + list(x[neg_list_col_name])), 
            axis=1
            )

    model.fillna('',inplace=True)

    return model



def drop_x_indices(model: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing or X indices
    """

    if 'X' in model.index or 'x' in model.index:
        logging.info('Dropping %s rows with X indices' % str(len(model.loc[['X']])))
        model.drop(['X'],axis=0,inplace=True)
    if '' in model.index:
        logging.info('Dropping %s rows missing indices' % str(len(model.loc[['']])))
        model.drop([''],axis=0,inplace=True)

    return model


def format_variable_names(model: pd.DataFrame) -> pd.DataFrame:
    """Format model variable names to make compatible with model checking
    """

    global _VALID_CHARS
    global _VAR_COL

    # remove whitespace in variable names
    model[_VAR_COL] = model[_VAR_COL].str.strip()

    # collect invalid element names in a list so they can be removed everywhere in the model
    # find invalid characters in element names and names starting with numbers
    invalid_names = [
        x for x in model[_VAR_COL] 
        if re.search(r'(^[0-9]+)',x.strip()) or re.search(r'([^'+_VALID_CHARS+']+)',x.strip())
        ]
    
    if len(invalid_names) > 0:
        logging.info('Formatting variable names: ')
    
    # remove invalid characters at the start of the variable name 
    replace_names = [re.sub(r'^[^'+_VALID_CHARS+']+','',x) for x in invalid_names]
    # replace invalid characters elsewhere in variable names
    replace_names = [re.sub(r'[^'+_VALID_CHARS+']+','_',x) for x in replace_names]
    
    # add ELE_ at the beginning of names starting with numbers
    replace_names = [re.sub(r'(^[0-9]+)','ELE_\\1',x) for x in replace_names]
    
    name_pairs = zip(invalid_names,replace_names)

    for (invalid_name,replace_name) in name_pairs:
        logging.info('%s -> %s' % (invalid_name,replace_name))
        model.replace(re.escape(invalid_name),re.escape(replace_name),regex=True,inplace=True)

    return model


def get_type(input_type):
    """Standardize element types
    """

    global _VALID_TYPES

    if input_type.lower() in _VALID_TYPES:
        return input_type
    elif input_type.lower().startswith('protein'):
        return 'protein'
    elif input_type.lower().startswith('chemical'):
        return 'chemical'
    elif input_type.lower().startswith('biological'):
        return 'biological'
    else:
        return 'other'


def model_to_dict(model: pd.DataFrame):
    """Convert model table to a dictionary
    """

    # convert dataframe to dict with variable name as the index
    model_dict = model.to_dict(orient='index')

    return model_dict


def model_from_dict(model_dict) -> pd.DataFrame:
    """Convert model dict back to dataframe
    """

    global _IDX_COL

    df_model = pd.DataFrame.from_dict(model_dict,orient='index')
    df_model.fillna('',inplace=True)
	# sort by number
    df_model.sort_values(by=[_IDX_COL],inplace=True)
    
    return df_model


def get_model_from_delphi(model_file: str) -> pd.DataFrame:

    global _IDX_COL

    ##############       adding the spreadsheet column names to a dataframe        ############
    column_names = [_IDX_COL,'Element Name', 'Element IDs', 'Element Type', 'Agent',
                    'Patient', 'Value Judgment', 'Specificity', 'Location', 'Time Scale / Frequency',
                    'Value: Activity / Amount ', 'Element NOTES', 'Variable', 'Positive',
                    'Negative','Influence Set NOTES', 'Levels', 'Spontaneous Behavior',
                    'Balancing Behavior', 'Update Group', 'Update Rate', 'Update Rank', 'Delay', 'Mechanism',
                    'Weight', 'Regulator Level', 'Evidence', 'Initial 0']
    df_model = pd.DataFrame(columns=column_names)

    ##############     Reading the json as a dict    ############
    with open(model_file) as json_file:
        data = json.load(json_file)

    json_data = pd.DataFrame.from_dict(data, orient='index').T

    ############      creating a list of the variables and adding them to the dataframe    ############
    variables_list = list()

    for var in json_data['variables'][0]:
        variables_list.append(var['name'])

    df_model['Variable'] = variables_list
    df_model[_IDX_COL] = [x+1 for x in range(len(variables_list))]

    ############    Reading the edges     ############
    positive_edges = {key: [] for key in variables_list}
    negative_edges = {key: [] for key in variables_list}
    evidence_for_edge = {key: [] for key in variables_list}

    for edge in json_data['edge_data'][0]:
        source = edge['source']
        target = edge['target']

        subj_polarities = list()
        obj_polarities = list()
        for evidence in edge['InfluenceStatements']:
            subj_polarities.append(evidence['subj_delta']['polarity'])
            obj_polarities.append(evidence['obj_delta']['polarity'])

        # if number of 1s = number of -1s, choose the first polarity
        subj_polarity = subj_polarities[0]
        obj_polarity = obj_polarities[0]

        # number of 1s != number of -1s, choose the most frequent polarity
        if subj_polarities.count(1) != subj_polarities.count(-1):
            subj_polarity = 1
            if subj_polarities.count(1) < subj_polarities.count(-1):
                subj_polarity = -1

        if obj_polarities.count(1) != obj_polarities.count(-1):
            obj_polarity = 1
            if obj_polarities.count(1) < obj_polarities.count(-1):
                obj_polarity = -1

        if subj_polarity == 1 and obj_polarity == 1:
            positive_edges[target].append(source)
        elif subj_polarity == -1 and obj_polarity == 1:
            positive_edges[target].append('!'+source)
        elif subj_polarity == 1 and obj_polarity == -1:
            negative_edges[target].append(source)
        elif subj_polarity == -1 and obj_polarity == -1:
            negative_edges[target].append('!'+source)

        evidence = edge['InfluenceStatements'][0]['evidence'][0]['text']
        evidence_for_edge[target].append(evidence)

    df_model['Positive'] = [
        ','.join(positive_edges[key]) for key in variables_list]
    df_model['Negative'] = [
        ','.join(negative_edges[key]) for key in variables_list]
    df_model['Evidence'] = [
        ','.join(evidence_for_edge[key]) for key in variables_list]
    df_model['Initial 0'] = 1

    df_model.fillna('',inplace=True)

    return df_model


def model_to_excel(model: pd.DataFrame, output_file: str, sheet_name='model'):
    """Save model to a file
    """

    global _VAR_COL

    model_out = copy.deepcopy(model)

    # remove regulator lists
    model_out.drop(columns=['Positive List','Negative List','Regulators'], inplace=True)

    model_out = model_out.reset_index().rename(columns={'index':_VAR_COL})
    model_out.to_excel(output_file, index=False, sheet_name=sheet_name)
    

def model_to_edges(model : pd.DataFrame) -> pd.DataFrame:
    """Convert the model into a dataframe of edges in the format
        element-regulator-interaction
    """
    
    # convert to dict for faster iteration
    model_dict = model_to_dict(model)

    edges_dict = dict()

    # create entries in edges_dict for each regulator-regulated pair in the model
    # using the model dict positive and negative regulator lists
    for key,item in model_dict.items():

        # re-parsing here to handle ! (not) notation
        # TODO: also handle AND, highest state, etc.
        pos_list = [x.strip() for x in re.findall(r'[a-zA-Z0-9\_!=]+',item.get('Positive',''))]
        neg_list = [x.strip() for x in re.findall(r'[a-zA-Z0-9\_!=]+',item.get('Negative',''))]

        # TODO: preserve element/regulator names and attributes
        pos_dict = {
            key+'pos'+str(i) : {'element':key, 'regulator':pos, 'interaction':'increases'}
            if pos[0]!='!' else {'element':key, 'regulator':pos[1:], 'interaction':'NOT increases'}
            for i,pos in enumerate(pos_list) 
            }
        neg_dict = {
            key+'neg'+str(i) : {'element':key, 'regulator':neg, 'interaction':'decreases'}
            if neg[0]!='!' else {'element':key, 'regulator':neg[1:], 'interaction':'NOT decreases'}
            for i,neg in enumerate(neg_list)
            }
        edges_dict.update(pos_dict)
        edges_dict.update(neg_dict)

    edges_df = pd.DataFrame.from_dict(edges_dict,orient='index')

    return edges_df


def model_to_edges_set(model : pd.DataFrame) -> set:
    """Convert the model into a set of edges in the format
        regulator-regulated-interaction with +/- for interaction
    """

    model_dict = model_to_dict(model)

    edges_set = set()

    # create entries in edges_dict for each regulator-regulated pair in the model
    # using the model dict positive and negative regulator lists
    for key,item in model_dict.items():

        # re-parsing here to handle ! (not) notation
        # TODO: also handle AND, highest state, etc.
        pos_list = re.findall(r'[a-zA-Z0-9\_!]+',item.get('Positive',''))
        neg_list = re.findall(r'[a-zA-Z0-9\_!]+',item.get('Negative',''))

        for i,pos in enumerate(pos_list):
            if pos[0]!='!':
                edges_set.add((pos, key, '+'))
            else:
                # NOT increases
                edges_set.add((pos[1:], key, '-')) 
        
        for i,neg in enumerate(neg_list):
            if neg[0]!='!':
                edges_set.add((neg, key, '-'))
            else:
                # NOT decreases
                edges_set.add((neg[1:], key, '+'))

    return edges_set


def model_to_networkx(model: pd.DataFrame) -> nx.DiGraph():
    """Convert model to a networkx graph
    """

    edges = model_to_edges(model)
    # In networkx 2.0 from_pandas_dataframe has been removed.
    graph = nx.from_pandas_edgelist(edges,
        source='regulator',target='element',edge_attr='interaction',
        create_using=nx.DiGraph())

    return graph


def model_to_sauce(model: pd.DataFrame, scenario=0, name_='', date_created='', 
    interventions=[], properties=[], parameters=[]) -> dict():
    """Convert model to json format (compatible with CRA's SAUCE)
    
        name : str 
        dateCreated : str 
        interventions : list of dicts with 'id', 'label', 'choices'
        properties : list of dicts with 'id', 'label', 'function'
        parameters (sensitivity analysis, inputs of interest, finding ‘pressure points’ in the system) :
            list of dicts with 'id', 'label', 'time', 'value'

    """

    # TODO: use inputs for interventions
    interventions = [{
        'id': 'Decision: food_aid', 
        'label': 'Simple Decision',
        'choices': [{'val': 0.0}, {'val': 1.0}, {'val': 2.0}]
    }]

    # TODO: convert property files
    properties = [{
        'id': 'Utility: food_security', 
        'label': 'Simple Utility',
        'function': {'operation': '', 'arguments': [{'ref':'food_security'}]}
    }]

    created_by = 'DySE Model Translator'

    model = model.reset_index()

    # TODO: use model_to_dict?
    input_col_X = [x.strip() for x in model.columns if ('variable' in x.lower())]
    input_col_A = [x.strip() for x in model.columns if ('positive' in x.lower() and x.lower() != 'positive list')]
    input_col_I = [x.strip() for x in model.columns if ('negative' in x.lower() and x.lower() != 'negative list')]
    input_col_initial = [
            x.strip() for x in model.columns 
            if ('initial' in x.lower() or 'scenario' in x.lower())
            ]
    input_col_name = [x.strip() for x in model.columns if ('element name' in x.lower())]
    input_col_ids = [x.strip() for x in model.columns if ('element ids' in x.lower())]
    input_col_type = [x.strip() for x in model.columns if ('element type' in x.lower())]

    # check for all required columns or duplicate colummns
    if (len(input_col_X) == 0) or (len(input_col_A) == 0) or (len(input_col_I) == 0) or (len(input_col_initial) == 0):
        raise ValueError('Missing one or more required columns in input file: variable, positive, negative, scenario')
    elif (len(input_col_X) > 1) or (len(input_col_A) > 1) or (len(input_col_I) > 1):
        raise ValueError('Duplicate column of Variable Name, Positive, or Negative')

    if (len(input_col_name) == 0) or (len(input_col_ids) == 0) or (len(input_col_type) == 0):
        raise ValueError('Missing one or more required column names: element name, element ids, element type')

    scenario_col = input_col_initial[scenario]
    var_col = input_col_X[0]
    pos_col = input_col_A[0]
    neg_col = input_col_I[0]
    name_col = input_col_name[0]
    type_col = input_col_type[0]

    model = model.rename(
        index=str,
        columns={
            var_col : 'id',
            name_col : 'label',
            type_col : 'description',
            scenario_col : 'initialValue'
        }
    )

    model['dtype'] = 'INT'
    model['function'] = model.apply(parse_influence_set,pos_col=pos_col,neg_col=neg_col,axis=1)    

    model_dict = model[['id','label','description','dtype','function','initialValue']].to_dict(orient='index')

    model_list = [ele_attr for key,ele_attr in model_dict.items()]

    model_out = defaultdict()
    model_out['name'] = name_
    model_out['dateCreated'] = date_created
    model_out['createdBy'] = created_by
    model_out['decisions'] = interventions
    model_out['utilities'] = properties
    model_out['parameters'] = parameters

    model_out['modelVariables'] = model_list

    return model_out

def model_to_delphi(model: pd.DataFrame, scenario=0, name_='', date_created='',
                    variables=[], timeStep="1.0", edge_data=[]) -> dict():
    """Convert model to json format (compatible with Delphi)
    

        name : str 
        dateCreated : str 
        variables : list of dicts with 'name', 'units', 'dtype','arguments', 'indicators'
        timeStep : str
        edge_data : list of dicts with 'source', 'target', 'CPT', 'polyfit', 'InfluenceStatements', 'evidence'


    """

    created_by = 'DySE Model Translator'

    model_dict = model_to_dict(model)
    variables = []
    edge_data = []
    for key, item in model_dict.items():
        # re-parsing here to handle ! (not) notation
        # TODO: also handle AND, highest state, etc.
        pos_list = re.findall(r'[a-zA-Z0-9\_!]+', item.get('Positive', ""))
        neg_list = re.findall(r'[a-zA-Z0-9\_!]+', item.get('Negative', ""))
        evidence = item.get('Evidence', "")

        arguments = []
        subj_polarity = []  # source polarity
        obj_polarity = []  # target polarity

        for i, pos in enumerate(pos_list):
            if pos[0] != '!':
                arguments.append(pos)
                subj_polarity.append(1)
                obj_polarity.append(1)
            else:
                arguments.append(pos[1:])
                subj_polarity.append(-1)
                obj_polarity.append(1)

        for i, neg in enumerate(neg_list):
            if neg[0] != '!':
                arguments.append(neg)
                subj_polarity.append(1)
                obj_polarity.append(-1)
            else:
                arguments.append(neg[1:])
                subj_polarity.append(-1)
                obj_polarity.append(-1)

        
        variables.append({
            "name": key,
            "units": "units",
            "dtype": "int",
            "arguments": arguments,
            "indicators": []
        })

        for j in range(len(arguments)):
            edge_data.append({
                "source": arguments[j],
                "target": key,
                "CPT": {},
                "polyfit": {},
                "InfluenceStatements": [{"type": "influence",
                                         "subj": {},
                                         "subj_delta": {
                                             "adjectives": [],
                                             "polarity":subj_polarity[j]},
                                         "obj":{},
                                         "obj_delta": {
                                             "adjectives": [],
                                             "polarity": obj_polarity[j]},
                                         "evidence": [{
                                             "source_api": "",
                                             "pmid": "",
                                             "text": evidence,
                                             "annotations": {
                                                 "found_by": "",
                                                 "provenance": []
                                             }
                                         }
                                         ],
                                         "id": "",
                                         "sbo": ""
                                         }]

            })

    dateCreated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_out = defaultdict()
    model_out['name'] = name_
    model_out['created_by'] = created_by
    model_out['dateCreated'] = dateCreated
    model_out['variables'] = variables
    model_out['timeStep'] = "1.0"
    model_out['edge_data'] = edge_data

    return model_out


def parse_influence_set(df_row,pos_col='positive',neg_col='negative'):
    """Parse influence set notation into nested functions
    """

    act_dict = rule_to_dict(df_row[pos_col])
    if act_dict is None:
        act_dict = {'val': 0}
    inh_dict = rule_to_dict(df_row[neg_col])
    if inh_dict is None:
        inh_dict = {'val': 0}

    # TODO: operator for this could be *, max, etc.
    influence_set_dict = {
        'operator' : '+', 
        'arguments' : [
            {'ref': df_row['id']},
            {'operator' : '-', 'arguments' : [inh_dict, act_dict]}
        ]
    }

    return influence_set_dict
    

def rule_to_dict(reg_rule, layer=0):

    op_key = 'operation'
    arg_key = 'arguments'
    ref_key = 'ref'
    and_func = 'MIN'
    or_func = 'MAX'
    not_func = 'NOT'
    sum_func = '+'

    if reg_rule:
        arguments = list()

        summation = False
        # TODO: include weights
        if reg_rule.find('+') == -1:
            reg_list = split_comma_outside_parentheses(reg_rule)
        else:
            if reg_rule.find(',') != -1:
                raise ValueError('Found mixed commas (OR) and plus signs (ADD) in regulator notation. Check for deprecated highest state notation element+ and replace with element^')
            else:
                reg_list = reg_rule.split('+')
                # set the summation flag to indicate these should be summed
                summation = True
        
        reg_list = [reg.strip() for reg in reg_list]

        for reg_element in reg_list:
            if reg_element[0]=='(' and reg_element[-1]==')':
                # AND operation
                arguments += [{
                    op_key : and_func,
                    arg_key : [
                        x for and_entity in split_comma_outside_parentheses(reg_element[1:-1])
                        for x in rule_to_dict(and_entity,1)
                        ] 
                    }]
            else:
                # single activator
                if reg_element[0]=='!':
                    arguments += [{
                        op_key : not_func,
                        arg_key : {ref_key : reg_element[1:]}
                        }] 
                else:
                    arguments += [{ref_key : reg_element}]

        if layer == 0:
            if summation:
                rule_dict = {op_key : sum_func, arg_key : arguments}
            else:
                rule_dict = {op_key : or_func, arg_key : arguments}

            return rule_dict
        else:
            return arguments


def split_comma_outside_parentheses(rule):
    final_list = list()
    parentheses = 0
    start = 0
    for index,char in enumerate(rule):
        if index==len(rule)-1:
            final_list.append(rule[start:index+1])
        elif char=='(' or char=='{' or char=='[':
            parentheses += 1
        elif char==')' or char=='}' or char==']':
            parentheses -= 1
        elif (char==',' and parentheses==0):
            final_list.append(rule[start:index])
            start = index+1
    return final_list
    

def main():
    parser = argparse.ArgumentParser(
        description='Process model files/objects and convert among formats.',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('input_file', type=str,
        help='Input model file name')
    parser.add_argument('output_file', type=str,
        help='Output model file name')
    parser.add_argument('--input_format', '-i', type=str, choices=['dyse','delphi'], 
        default='dyse',
        help='Input file format \n'
        '\t dyse (default): DySE model tabular format \n'
        '\t delphi: json format from Delphi \n')
    parser.add_argument('--output_format', '-o', type=str, choices=['dyse','json','edges','sauce','delphi'], 
        default='dyse',
        help='Output file format \n'
        '\t dyse (default): DySE model tabular format \n'
        '\t json: json format from model dictionary \n'
        '\t edges: element-regulator-interaction triplets \n'
        '\t sauce: CRA json format for SAUCE analysis \n'
        '\t delphi: json format compatible with Delphi \n')

    args = parser.parse_args()

    # set up logging to stdout
    # TODO: define standard logger
    log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    log_handler = logging.StreamHandler(sys.stdout)
    log_handler.setFormatter(log_formatter)
    logger = logging.getLogger()
    logger.addHandler(log_handler)
    logger.setLevel(logging.DEBUG)

    if args.input_format == 'delphi':
        model = get_model_from_delphi(args.input_file)
    else:
        model = get_model(args.input_file)

    # save model in specified output format
    if args.output_format == 'dyse' and args.input_format != 'delphi':
        model_to_excel(model, args.output_file)

    elif args.output_format == 'json':
        model_dict = model_to_dict(model)
        with open(args.output_file,'w') as json_out:
            json.dump(model_dict, json_out) 

    elif args.output_format == 'edges':
        edges = model_to_edges(model)
        edges.to_excel(args.output_file, index=False)

    elif args.output_format == 'sauce':
        sauce_model = model_to_sauce(model,name_=args.input_file)
        with open(args.output_file,'w') as json_out:
            json.dump(sauce_model, json_out)
    
    elif args.output_format == 'delphi':
        delphi_model = model_to_delphi(model,name_=args.input_file)
        with open(args.output_file,'w') as json_out:
            json.dump(delphi_model, json_out, indent=1,
                      separators=(',', ':'), default=str)
    
    elif args.output_format == 'dyse' and args.input_format == 'delphi':
        model.to_excel(args.output_file, index=False)

    else:
        raise ValueError('Invalid output format')

if __name__ == '__main__':
    main()
