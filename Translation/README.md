# Translation

## Description of files

- `interactions.py` - functions for processing interactions and reading from files (preliminaries for a separate class)
- `model.py` - functions for processing models and reading from files (preliminaries for a separate class)

## Usage

- `model.py`
~~~
usage: model.py [-h] [--output_format {dyse,json,edges,sauce}]
                input_file output_file

Process model files/objects and convert among formats.

positional arguments:
  input_file            Input model file name
  output_file           Output model file name

optional arguments:
  -h, --help            show this help message and exit
  --output_format {dyse,json,edges,sauce}, -o {dyse,json,edges,sauce}
                        Output file format 
                        	 dyse (default): DySE model tabular format 
                        	 json: json format 
                        	 edges: element-regulator-interaction triplets 
                        	 sauce: CRA json format for SAUCE analysis 
~~~

- `interactions.py`
~~~
usage: interactions.py [-h] [--model_file MODEL_FILE]
                       [--output_format {ic,schema,mitre,bio}] [--compare]
                       [--score_threshold SCORE_THRESHOLD] [--protein_only]
                       input_files output_file

Process interaction files/objects and convert among formats.

positional arguments:
  input_files           interactions in tabular format. 
                        can be a directory of files or multiple comma-separated names. 
                        supported file formats: excel (SCHEMA or PosReg/NegReg format), csv (IC output), dms (IC output)
  output_file           Output file name

optional arguments:
  -h, --help            show this help message and exit
  --model_file MODEL_FILE, -m MODEL_FILE
                        model file to map element and variable names
  --output_format {ic,schema,mitre,bio}, -o {ic,schema,mitre,bio}
                        format of interactions output 
                        	 ic (default): same format as interaction classifier output, compatible with extension 
                        	 schema: all DySE column names (Element, Regulator and attributes) 
                        	 mitre: abbreviated MITRE tabular format 
                        	 bio: PosReg NegReg column format (for input to interaction classifier)
  --compare, -c         Keep file index to compare interactions among files
  --score_threshold SCORE_THRESHOLD, -s SCORE_THRESHOLD
                        threshold score to filter interactions
  --protein_only, -p    use only protein-protein interactions
~~~

## Examples
- see [`examples/test-translation.bash`](examples/test-translation.bash)