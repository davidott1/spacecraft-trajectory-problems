import sys
import json
import argparse
from src.load.parser import parse_command_line_interface_input
from src.load.configurer import configure_output

def parse_read_configure():
    """
    General function to configure input and output folder and read input parameters
    - filenpath of the input
    - folderpath of the output
    """ 

    # Parse file and folderpath
    parsed_cli_input = parse_command_line_interface_input()

    # Unpack parsed paths
    input_filepath_input    = parsed_cli_input['input_filepath_input']
    output_folderpath_input = parsed_cli_input['output_folderpath_input']

    # Read input parameters
    (
        input_filepath, 
        input_parameters,
    ) = \
        read_input_parameters(
            input_filepath_input,
        )

    # Configure output folderpath
    output_folderpath = \
        configure_output(
            output_folderpath_input,
        )

    # Pack up
    return {
        'input_filepath'    : input_filepath   ,
        'output_folderpath' : output_folderpath,
        'input_parameters'  : input_parameters ,
    }


def read_input_parameters(
        input_filepath_input,
    ):
    use_json_reader = False
    if input_filepath_input.endswith('.json'):
        input_filepath = input_filepath_input
        use_json_reader = True
    else:
        input_filepath = f"{input_filepath_input}.json"
        use_json_reader = True
    if use_json_reader:
        input_parameters = read_json(input_filepath)
        print(f"    Input Filepath    : {input_filepath}")
    return (
        input_filepath, 
        input_parameters,
    )


def read_json(filename):
    
    with open(filename, "r") as file:
       parameters_input = json.load(file)

    return parameters_input