import os
import sys
import pandas as pd
import argparse
import logging
import networkx as nx
from typing import List

from Translation.model import get_model, get_type, model_to_dict

# TODO: Make into a class definition for interactions object

# Define column names for DySE format
_COLNAMES = [
    # Provenance attributes
    'Source', 'Reader', 'Evidence', 'Evidence Index', 'Notes',
    # Element variable and attributes
    'Element Variable',
    'Element Name', 'Element Text', 'Element Database', 'Element ID', 'Element Type',
    'Element Agent', 'Element Patient',
    'Element ValueJudgment', 'Element Value Type', 'Element Scope',
    'Element Level', 'Element Change', 'Element Degree',
    'Element Location', 'Element Timing',
    # Interaction function and attributes
    'Interaction Function', 
    'Interaction Name', 'Interaction Text', 'Interaction ID', 'Interaction Type', 
    'Interaction Mechanism', 'Interaction Degree',
    'Interaction Location', 'Interaction Timing',
    # Regulator variable and attributes
    'Regulator Variable',
    'Regulator Name', 'Regulator Text', 'Regulator Database', 'Regulator ID', 'Regulator Type',
    'Regulator Agent', 'Regulator Patient',
    'Regulator ValueJudgment', 'Regulator Value Type', 'Regulator Scope',
    'Regulator Level', 'Regulator Change', 'Regulator Degree',
    'Regulator Location', 'Regulator Timing',
    # Scoring metrics
    'Reader Count', 'Source Count', 'Evidence Count',
    'Total Score', 'Kind Score', 'Match Level', 'Epistemic Value', 'Belief',
    ]

# columns to merge duplicate values
_DUP_COLS = [
    'Regulator Name', 'Regulator ID', 'Regulator Type', 'Regulator Location',
    'Element Name', 'Element ID', 'Element Type', 'Element Location',
    'Interaction Function'
    ]
_SOURCE_COLS = ['Source', 'Reader', 'Evidence']
_SCORE_COLS = ['Total Score', 'Kind Score', 'Match Level', 'Epistemic Value', 'Belief',
    'Reader Count', 'Source Count', 'Evidence Count'
    ]


# Define regex for processing names
_REPLACE_CHARS = r'[\ \-,\/]+'
_REMOVE_CHARS = r'[^a-zA-Z0-9\_]+'

def get_interactions(input_file: str) -> pd.DataFrame:
    """Get interactions from a file, detect file format, and map to DySE schema
    """

    file_type = os.path.splitext(input_file)[-1]

    if file_type in ['.xls','.xlsx']:
        # check for model file or interactions file
        df_input = pd.read_excel(
                input_file, 
                sheet_name = 0, 
                na_values='NaN', 
                keep_default_na=False
                )
        if ('element attributes' in [x.lower() for x in df_input.columns]
                or 'variable' in [x.lower() for x in df_input.columns]
                ):
            # model file
            # TODO: need to disassemble model and get names and IDs rather than just variable names returned by model_to_edges
            # model = get_model(input_file)
            # interactions = model_to_edges(model)
            raise ValueError('Model file input not yet supported')
        else:
            # interactions file
            interactions = df_input
            if 'PosReg Name' in interactions.columns:
                interactions_format = 'bio'
            elif 'Regulator Name' in interactions.columns:
                interactions_format = 'schema'
            else:
                raise ValueError('Unrecognized file format')
    elif file_type in ['.csv','.dms']:
        interactions = pd.read_csv(
                    input_file, 
                    na_values='NaN', 
                    keep_default_na=False, 
                    dtype=str
                    )
        if 'regulator_name' in interactions.columns:
            interactions_format = 'ic'
        else:
            raise ValueError('Unrecognized file format')
    elif file_type in ['.tsv']:
        interactions = pd.read_csv(
                input_file, 
                sep='\t',
                na_values='NaN', 
                keep_default_na=False, 
                dtype=str
                )
        if 'PosReg Name' in interactions.columns:
            interactions_format = 'bio'
        else:
            raise ValueError('Unrecognized file format')
    else:
        raise ValueError(
                'Unrecognized file type, '
                'must be csv, dms, tsv, or Excel: {}'.format(file_type)
                )

    schema = pd.DataFrame(columns=_COLNAMES)

    if interactions_format == 'ic':
        schema['Element Name'] = interactions['regulated_name']
        schema['Element Database'] = interactions['database2']
        schema['Element ID'] = interactions['ID2']
        schema['Element Type'] = interactions['element_type2']
        schema['Element Location'] = interactions['location2']
        schema['Element Level'] = interactions['level2']
        schema['Regulator Name'] = interactions['regulator_name']
        schema['Regulator Database'] = interactions['database1']
        schema['Regulator ID'] = interactions['ID1']
        schema['Regulator Type'] = interactions['element_type1']
        schema['Regulator Location'] = interactions['location1']
        schema['Regulator Level'] = interactions['level1']
        schema['Interaction Function'] = interactions['interaction']
        schema['Total Score'] = pd.to_numeric(interactions['score'])
        schema['Kind Score'] = pd.to_numeric(interactions['kindscore'])
        schema['Match Level'] = pd.to_numeric(interactions['matchlevel'])
        schema['Epistemic Value'] = pd.to_numeric(interactions['epistemicvalue'])
        if 'evidence' in interactions.columns:
            schema['Evidence'] = interactions['evidence']
        # get only unique file numbers
        schema['Source'] = interactions['file_numbers'].apply(
                lambda x: ';'.join(set(x.split(' ')))
                )
        schema['Source Count'] = schema['Source'].apply(
                lambda x: len(x.split(';'))
                )
        schema['Evidence Count'] = pd.to_numeric(interactions['number'])

    elif interactions_format == 'edges':
        raise ValueError('Input format not yet supported')
    elif interactions_format == 'bio':
        raise ValueError('Input format not yet supported')
        # add interaction column
        # interactions = interactions.apply(get_interaction_from_reg_col,axis=1)
        # TODO: duplicate rows with both posreg and negreg
        # if interactions['Interactions']
        # element_col = 'Element Name'
        # regulator_col = ''
        raise ValueError('Input format not yet supported')
    elif interactions_format == 'schema':
        schema = interactions
    else:
        raise ValueError('Unrecognized input format')

    schema.fillna('',inplace=True)

    # TODO: call process/format_interactions here?
        
    return schema


def merge_interactions(
        input_files: List[str], 
        text_names=False, 
        keep_hanging_nodes=False
        ) -> pd.DataFrame:
    """Load and merge interaction files, adding an index to keep track of the
        source file
    """

    df_interactions = list()
    file_index = 1
    for file_ in input_files:
        df_temp = get_interactions(file_)
        df_temp = format_interactions(df_temp, text_names, keep_hanging_nodes)
        df_temp['Indices'] = str(file_index)
        df_interactions.append(df_temp)
        file_index += 1

    df_merged = df_interactions[0]
    for df_ in df_interactions[1:]:
        df_merged = df_merged.merge(df_, how='outer')

    return df_merged


def format_interactions(
        interactions: pd.DataFrame, 
        text_names=False,
        keep_hanging_nodes=False
        ) -> pd.DataFrame:
    """Remove invalid characters and standardize values in interactions columns
    """

    global _REPLACE_CHARS
    global _REMOVE_CHARS

    # fill missing IDs with text
    interactions = interactions.apply(fill_missing_ids, axis=1)

    if text_names:
        interactions['Regulator Name'] = interactions['Regulator Text']
        interactions['Element Name'] = interactions['Element Text']

    if not keep_hanging_nodes:
        # remove rows with missing source or target elements 
        interactions = interactions[
            (interactions['Regulator Name'] != '') 
            & (interactions['Element Name'] != '')
            ]

    # process invalid characters
    # -> replace _REPLACE_CHARS characters with _
    # -> remove _REMOVE_CHARS
    # -> add ELE_ at the beginning of names starting with numbers
    for col in ['Regulator Name','Element Name']:
        interactions[col] = interactions[col].str.replace(
                _REPLACE_CHARS, '_'
                ).str.replace(
                _REMOVE_CHARS, ''
                ).str.replace(
                r'(^[0-9]+)','ELE_\\1'
                ).str.lower()

    # standardize types
    interactions['Regulator Type'] = \
            interactions['Regulator Type'].apply(get_type)
    interactions['Element Type'] = \
            interactions['Element Type'].apply(get_type)

    # get interaction function from source/target change/level
    interactions['Interaction Function'] = \
            interactions.apply(get_function, axis=1)

    interactions.fillna('', inplace=True)

    return interactions


def score_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    """Calculate scores (not yet implemented, currently inserts placeholders)
    """

    global _DUP_COLS
    global _SCORE_COLS
    global _SOURCE_COLS

    final = pd.DataFrame(columns=interactions.columns)

    # define columns to count unique values for scoring 

    agg_unique_cols = list(interactions.columns)
    [agg_unique_cols.remove(x) for x in _DUP_COLS + _SOURCE_COLS]

    # TODO: detect contradictions __within the reading__ 
    interactions = interactions.groupby(_DUP_COLS).agg({
        **{col : ['unique'] for col in agg_unique_cols},
        **{col : ['unique','nunique'] for col in _SOURCE_COLS}
        })

    interactions.reset_index(inplace=True)

    # write to final output table
    # copy dup_cols columns directly
    for col in _DUP_COLS:
        final[col] = interactions[col]
    
    # concatenate values in aggregate lists for source_cols
    for col in agg_unique_cols + _SOURCE_COLS:
        final[col] = interactions[(col, 'unique')].map(
            lambda x: '; '.join([str(a) for a in x if str(a)!=''])
            )
    
    # count unique values for source counts
    for col in _SOURCE_COLS:
        final[col+' Count'] = interactions[(col,'nunique')]

    # classification scoring metrics
    # TODO: get kind score and match score using classifier code
    # TODO: label contradiction/extension/corroboration in addition to the kind score
    # TODO: label specifying, weak, etc in addition to score
    # final = classify(final)
    logging.warn('Kind Score, Match Level, Epistemic Value calculation '
            'not yet implemented, inserting placeholder values'
            )
    # kind score : relation to baseline model 
    # (extension, contradiction, corroboration)
    final['Kind Score'] = 0
    # match level : location, type, etc.
    final['Match Level'] = 1
    # epistemic value : confidence from reading
    final['Epistemic Value'] = 1
    final['Total Score'] = final.apply(get_score,axis=1)

    return final


def process_interactions(
        interactions: pd.DataFrame, 
        score_threshold=0, 
        keep_file_index=False, 
        protein_only=False, 
        keep_dup_scores=False,
        invalid_regulators=[],
        invalid_elements=[]
        ) -> pd.DataFrame:
    """Merge duplicates and count sources, databases, repetitions
    """

    global _DUP_COLS
    global _SCORE_COLS
    global _SOURCE_COLS

    # filter by score
    if score_threshold > 0:
        interactions = interactions.loc[interactions['Total Score'] >= 10]
    
    # filter for protein-protein interactions
    if protein_only:
        interactions = interactions[
            interactions['Element Type'].str.contains('protein')
            & interactions['Regulator Type'].str.contains('protein')
            ]
    
    if invalid_regulators != [] or invalid_elements != []:
        interactions = interactions[
            ~interactions['Element Name'].isin(invalid_elements)
            & ~interactions['Regulator Name'].isin(invalid_regulators)
            ]
    
    # check for missing scores
    if interactions['Total Score'].any() == '':
        interactions = score_interactions(interactions)
    
    # merge across duplicates 
    # take the row with the max total score across duplicates
    interactions_grouped = interactions.loc[
            interactions.groupby(_DUP_COLS)['Total Score'].idxmax()
            ]

    if ((keep_file_index and 'Indices' in interactions.columns) 
            or keep_dup_scores
            ):
        # temporarily set index to attach grouped columns
        interactions_grouped.set_index(_DUP_COLS,inplace=True)

        if keep_file_index:
            # concatenate file indices if comparing interaction sets
            grouped_indices = interactions.groupby(_DUP_COLS)['Indices'].agg(
                    lambda x: ';'.join(x)
                    )
            interactions_grouped['Indices'] = grouped_indices

        if keep_dup_scores:
            grouped_scores = interactions.groupby(_DUP_COLS)[_SCORE_COLS].agg(
                    lambda x: ';'.join([str(a) for a in x])
                    )
            for col in _SCORE_COLS:
                interactions_grouped['All '+col] = grouped_scores[col]
        
        interactions_grouped.reset_index(inplace=True)

    interactions_grouped.fillna('',inplace=True)

    # sort output columns
    sort_cols = _SCORE_COLS + _SOURCE_COLS + [
            'Element Name', 'Element ID', 'Regulator Name', 'Regulator ID'
            ]
    sort_orders = [False for x in _SCORE_COLS] \
        + [False for x in _SOURCE_COLS] \
        + [True, True, True, True]

    interactions_grouped = interactions_grouped.sort_values(
            sort_cols,ascending=sort_orders
            )

    return interactions_grouped


def get_function(df_row):
    """Use regulator/regulated changes/levels to determine interaction function
        increase x -> increase y => positive interaction (x increases y)
        increase x -> decrease y => negative interaction (x decreases y)
        decrease x -> increase y => invert positive interaction (!x increases y)
        decrease x -> decrease y => invert negative interaction (!x decreases y)
    """

    # TODO: use increase/decrease separately from levels
    increase_terms = ['','increase','inc','positive','active','1','none']
    decrease_terms = ['decrease','decreased','dec','-1']
    low_level_terms = ['low','little','inactive']

    if 'Interaction ID' in df_row:
        interaction_type = df_row['Interaction ID'].lower()
    else:
        interaction_type = ''

    if 'Regulator Change' in df_row:
        reg_change = df_row['Regulator Change'].lower()
    else:
        reg_change = ''

    if 'Element Change' in df_row:
        ele_change = df_row['Element Change'].lower()
    else:
        ele_change = ''
    
    reg_degree = df_row['Regulator Level'].lower()
    ele_degree = df_row['Element Level'].lower()

    if not(reg_change in increase_terms + decrease_terms):
        logging.warn('Unrecognized Regulator Change: %s' % reg_change)
    if not(ele_change in increase_terms + decrease_terms):
        logging.warn('Unrecognized Element Change: %s' % ele_change)
    
    if (interaction_type == 'preventrelation' 
            or (ele_change in decrease_terms or ele_degree in low_level_terms)
            ):
        interaction_change = 'decreases'
    else:
        interaction_change = 'increases'

    if (reg_change in decrease_terms or reg_degree in low_level_terms):
        interaction_function = 'NOT ' + interaction_change
    else:
        interaction_function = interaction_change

    return interaction_function

def fill_missing_ids(df_row):
    """Look for missing IDs and fill with names
    """

    if df_row['Regulator ID'] == '':
        df_row['Regulator ID'] = df_row['Regulator Name']
    
    if df_row['Element ID'] == '':
        df_row['Element ID'] = df_row['Element Name']

    return df_row


def map_extensions(
        interactions: pd.DataFrame, 
        model: pd.DataFrame
        ) -> pd.DataFrame:
    """Map model names and variables to interactions
    """

    interactions = interactions.apply(get_model_names,df_model=model,axis=1)
    interactions = interactions.apply(get_variable_names,df_model=model,axis=1)

    return interactions


def get_model_names(df_row, df_model):
    """Iterate through model synonyms/IDs to find a match for the input ID.
        If a match is found, return the model name, otherwise keep the existing 
        name.
    """

    reg_id = df_row['Regulator ID'].lower().strip()
    ele_id = df_row['Element ID'].lower().strip()

    # TODO: speed up this iteration
    for model_index, model_row in df_model.iterrows():

        model_ids = [
                str(x).lower().strip() 
                for x in model_row['Element IDs'].split(',')
                ]

        if reg_id in model_ids:
            df_row['Regulator Name'] = model_row['Element Name']
        
        if ele_id in model_ids:
            df_row['Element Name'] = model_row['Element Name']

    return df_row


def get_variable_names(
        df_row, 
        df_model, 
        label_extensions=False,
        confidence_threshold = 2
        ):
    """Match interaction elements to variable names in a model 
    """

    model_id_key = 'Element IDs'
    model_name_key = 'Element Name'
    model_type_key = 'Element Type'

    model_dict = model_to_dict(df_model)

    for element in ['Regulator','Element']:

    	# TODO: include location and type in generated variable name 
        variable = df_row[element + ' Name'] 
        if label_extensions:
            variable += '_ext'
        
        element_id = df_row[element + ' ID'].upper()
        element_name = df_row[element + ' Name'].upper() 
        element_type = df_row[element+' Type'].upper()    

        # Iterate all names in the dictionary and find the most likely match
        # TODO: standardize this matching with the IC
        previous_confidence = 0.0
        for model_variable,model_var_attributes in model_dict.items():

            current_confidence = 0.0
            
            model_variable_name = model_var_attributes[model_name_key].upper()
            model_variable_type = model_var_attributes[model_type_key].upper()

            # check for matching IDs or names
            if (element_id in model_var_attributes[model_id_key]):
                current_confidence = 1
            elif (element_name.startswith(model_variable_name) 
                    or model_variable_name.startswith(element_name)
                    ):
                current_confidence = 0.8

            # check for matching types
            if (current_confidence > 0 
                    and model_variable_type.startswith(element_type)
                    ):
                current_confidence += 1

            if current_confidence > previous_confidence:
                variable = model_variable
                previous_confidence = current_confidence
                if current_confidence==confidence_threshold: 
                    break

        df_row[element + ' Variable'] = variable
    
    return df_row


def get_score(df_row):
    """Calculate overall score based on individual classification scores
    """
    # FIXME: use classifier score calculation
    score = (
            (int(df_row['Kind Score']) + int(df_row['Match Level'])
            ) 
            * int(df_row['Epistemic Value'])
            )

    return score


def interactions_to_ic(interactions: pd.DataFrame) -> pd.DataFrame:
    """Convert DySE schema to Interaction Classifier (IC) output format.

        For compatibility with model extension. TODO: check if it can be removed
    """

    ic_col_names = [
            'regulator_name','regulator_id','database1','ID1','location1',
            'element_type1','cell_type1','level1',
            'regulated_name','regulated_id','database2','ID2','location2',
            'element_type2','cell_type2','level2',
            'interaction','score','kindscore','matchlevel',
            'epistemicvalue','file_numbers','number'
            ]

    final = pd.DataFrame(columns=ic_col_names)
    
    final['regulator_name'] = interactions['Regulator Name']
    final['regulated_name'] = interactions['Element Name']

    for ic_col in [
            'regulator_id', 
            'database1', 'cell_type1', 
            'database2', 'cell_type2'
            ]:
        if ic_col in interactions.columns:
            final[ic_col] = interactions[ic_col]
    
    final['ID1'] = interactions['Regulator ID']
    final['ID2'] = interactions['Element ID']

    final['location1'] = interactions['Regulator Location']
    final['location2'] = interactions['Element Location']

    final['element_type1'] = interactions['Regulator Type']
    final['element_type2'] = interactions['Element Type']

    final['level1'] = interactions['Regulator Level']
    final['level2'] = interactions['Element Level']

    final['database1'] = interactions['Regulator Database']
    final['database2'] = interactions['Element Database']

    final['interaction'] = interactions['Interaction Function']
    
    final['file_numbers'] = interactions['Source']
    
    final['number'] = interactions['Evidence Count']
    
    final['kindscore'] = interactions['Kind Score']
    final['matchlevel'] = interactions['Match Level']
    final['epistemicvalue'] = interactions['Epistemic Value']

    final['score'] = interactions['Total Score']

    if 'Indices' in interactions.columns:
        final['indices'] = interactions['Indices']

    final = final.sort_values(
        ['score','number','regulator_name','ID1','regulated_name','ID2'],
        ascending=[False,False,True,True,True,True]
        )

    final.fillna('')

    return final


def interactions_to_mitre(interactions: pd.DataFrame) -> pd.DataFrame:
    """Convert DySE schema to MITRE abbreviated tabular format
    """
    
    mitre_colnames = [
            'Source',	'System', 'Sentence ID',
	        'Factor A Text', 'Factor A Normalization', 
            'Factor A Modifiers', 'Factor A Polarity',	
            'Relation Text', 'Relation Normalization',	'Relation Modifiers',
            'Factor B Text', 'Factor B Normalization',	
            'Factor B Modifiers', 'Factor B Polarity',
            'Location',	'Time',	'Evidence'
            ]

    final = pd.DataFrame(columns=mitre_colnames)

    # column mapping
    final['Source'] = interactions['Source']
    final['System'] = interactions['Database']
    final['Sentence ID'] = interactions['Evidence Index']

    final['Factor A Text'] = interactions['Regulator Text']
    final['Factor A Normalization'] = interactions['Regulator ID']
    final['Factor A Modifiers'] = \
            interactions[['Regulator Degree','Regulator Level']].apply(
            lambda x: '; '.join([a for a in x if a!='']),
            axis=1
            )
    final['Factor A Polarity'] = \
            interactions[['Regulator Change']].apply(
            lambda x: '; '.join([a for a in x if a!='']),
            axis=1
            )
    final['Relation Text'] = interactions['Interaction Text']
    final['Relation Normalization'] = interactions['Interaction ID']	
    final['Relation Modifiers'] = interactions['Interaction Degree']
    final['Factor B Text'] = interactions['Element Text']
    final['Factor B Normalization'] = interactions['Element ID']
    final['Factor B Modifiers'] = \
            interactions[['Element Degree','Element Level']].apply(
            lambda x: '; '.join([a for a in x if a!='']), 
            axis=1
            )
    final['Factor B Polarity'] = \
            interactions[['Element Change']].apply(
            lambda x: '; '.join([a for a in x if a!='']), 
            axis=1
            )
    
    # mitre format doesn't have space for separate regulator/element 
    # locations or timing, so concatenate them all here
    final['Location'] = \
            interactions[
                    ['Interaction Location','Regulator Location','Element Location']
                    ].apply(
                    lambda x: '; '.join([a for a in x if a!='']), axis=1
                    )
    final['Time'] = \
            interactions[
                    ['Interaction Timing','Regulator Timing','Element Timing']
                    ].apply(
                    lambda x: '; '.join([a for a in x if a!='']), axis=1
                    )
    final['Evidence'] = interactions['Evidence']
    final['Reader Count'] = interactions['Reader Count']
    final['Source Count'] = interactions['Source Count']
    final['Evidence Count'] = interactions['Evidence Count']

    final.fillna('')

    return final


def interactions_to_networkx(
        interactions: pd.DataFrame, 
        use_variables=True
        ) -> nx.DiGraph():
    """Convert interactions to a NetworkX directed graph
    """

    # TODO: include more element/regulator attributes
    
    if use_variables:
        source = 'Regulator Variable'
        target = 'Element Variable'
    
    else:
        source = 'Regulator Name'
        target = 'Element Name'
    
    interactions_renamed = interactions.rename(
            index=str, 
            columns={source : 'regulator', 
                     target : 'element',
                     'Interaction Function' : 'interaction',
                     'Total Score' : 'weight'
                    }
           )

    # NOTE: In networkx 2.0 from_pandas_dataframe has been removed
    graph = nx.from_pandas_edgelist(interactions_renamed,
        source='regulator', 
        target='element', 
        edge_attr=['interaction','weight'],
        create_using=nx.DiGraph())

    return graph


def main():
    parser = argparse.ArgumentParser(
        description='Process interaction files/objects and convert among formats.',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('input_files', type=str,
                        help='interactions in tabular format. \n'
                        'can be a directory of files or multiple comma-separated names. \n'
                        'supported file formats: excel (SCHEMA or PosReg/NegReg format), csv (IC output), dms (IC output)')
    parser.add_argument('output_file', type=str,
                        help='Output file name')
    parser.add_argument('--model_file', '-m', type=str, default='',
                        help='model file to map element and variable names')
    parser.add_argument('--output_format', '-o', type=str, choices=['ic','schema','mitre','bio'], 
                        default='schema',
                        help='format of interactions output \n'
                        '\t ic (default): same format as interaction classifier output, compatible with extension \n'
                        '\t schema: all DySE column names (Element, Regulator and attributes) \n'
                        '\t mitre: abbreviated MITRE tabular format \n'
                        '\t bio: PosReg NegReg column format (for input to interaction classifier)' 
                        )
    parser.add_argument('--compare', '-c', action='store_true',
                        help='Keep file index to compare interactions among files')
    parser.add_argument('--score_threshold', '-s', type=int, default=0,
                        help='threshold score to filter interactions')
    parser.add_argument('--keep_dup_scores', action='store_true',
                        help='keep all scores for duplicate rows instead of taking the max score')
    parser.add_argument('--protein_only', '-p', action='store_true',
                        help='use only protein-protein interactions')
                        # TODO: warn if bio format and not filtering for only protein-protein

    args = parser.parse_args()

    # set up logging to stdout
    # TODO: define standard logger
    log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    log_handler = logging.StreamHandler(sys.stdout)
    log_handler.setFormatter(log_formatter)
    logger = logging.getLogger()
    logger.addHandler(log_handler)
    logger.setLevel(logging.DEBUG)

    if os.path.isdir(args.input_files):
        # get all files of supported types in the directory, ignore temp files that start with '~$'
        input_files = []
        for ext in ['.xlsx','.xls','.csv','.dms']:
            input_files.extend(glob.glob(os.path.join(args.input_files,'*'+ext)))
            
        input_files = [x for x in input_files if '$' not in x]
    else:
        input_files = args.input_files.split(',')
    
    interactions = merge_interactions(input_files, args.model_file)
    interactions = format_interactions(interactions)
    interactions = process_interactions(
        interactions, score_threshold=args.score_threshold, 
        keep_file_index=args.compare, protein_only=args.protein_only,
        keep_dup_scores=args.keep_dup_scores)

    # format output
    if args.output_format =='schema' or args.output_format == 'mitre':
        # check output file extension
        if os.path.splitext(args.output_file)[-1] not in ['.xls','.xlsx']:
            output_file = os.path.splitext(args.output_file)[0] + '.xlsx'
            logging.info('Corrected output file extension: %s' % output_file)
        else:
            output_file = args.output_file
        
        if args.output_format == 'mitre':
            # convert to mitre format
            interactions_mitre = interactions_to_mitre(interactions)
            interactions_mitre.to_excel(output_file,index=None)
        else:
            # default format
            interactions.to_excel(output_file,index=None)

    elif args.output_format == 'ic':
        # check output file extension
        if os.path.splitext(args.output_file)[-1] not in ['.csv']:
            output_file = os.path.splitext(args.output_file)[0] + '.csv'
            logging.info('Corrected output file extension: %s' % output_file)
        else:
            output_file = args.output_file
        
        # convert to ic output format
        interactions_ic = interactions_to_ic(interactions)
        interactions_ic.to_csv(output_file,index=None)

    else:
        raise ValueError('Unrecognized output format')


if __name__ == '__main__':
    main()