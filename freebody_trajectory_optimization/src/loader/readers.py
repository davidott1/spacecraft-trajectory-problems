import sys
import json

def configure_files_folders():
    """
    General function to configure input and output files and folders
    - filenpath of the input
    - folderpath of the output
    """ 
    print("\nConfiguring and Reading Files and Folders")

    # Grab command line input
    input_filepath_input    = sys.argv[1]
    output_folderpath_input = sys.argv[2]

    # Setting reader booleans
    use_json_reader = False

    # Check input filename extension
    if input_filepath_input.endswith('.json'):
        input_filepath = input_filepath_input
        use_json_reader = True
    else:
        input_filepath = f"{input_filepath_input}.json"
        use_json_reader = True
    
    # Use particular reader
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