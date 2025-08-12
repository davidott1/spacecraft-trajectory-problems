import sys
import json
import argparse

def configure_and_read():
    """
    General function to configure input and output folder and read input parameters
    - filenpath of the input
    - folderpath of the output
    """ 

    # Configure input and output folders
    print("\nConfigure Input and Output Folders")
    parser = argparse.ArgumentParser(description="Configure input and output files and folders")
    parser.add_argument(
        "input_filepath"                          ,
        help            = "Path to the input file",
    )
    parser.add_argument(
        "output_folderpath"                                    ,
        default = "output"                                     ,
        nargs   = "?"                                          ,
        help    = "Path to the output folder (default: output)",
    )
    args                    = parser.parse_args()
    input_filepath_input    = args.input_filepath
    output_folderpath_input = args.output_folderpath

    # Read input parameters
    use_json_reader = False

    if input_filepath_input.endswith('.json'):
        input_filepath = input_filepath_input
        use_json_reader = True
    else:
        input_filepath = f"{input_filepath_input}.json"
        use_json_reader = True
    
    if use_json_reader:
        input_parameters = read_json(input_filepath)

    # Output folderpath
    output_folderpath = output_folderpath_input # do nothing for now
    print(f"  Output : {output_folderpath}")

    # Pack up
    return {
        'input_filepath'    : input_filepath   ,
        'output_folderpath' : output_folderpath,
        'input_parameters'  : input_parameters ,
    }

def read_json(filename):
    
    with open(filename, "r") as file:
       parameters_input = json.load(file)
    print(f"  Input  : {filename}")

    return parameters_input