import argparse

def parse_command_line_interface_input():
    """
    General function to configure input and output folder and read input parameters
    - filenpath of the input
    - folderpath of the output
    """ 

    # Parse file and folderpath
    print("\nParse Filepaths and Folderpaths")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_filepath"               ,
        help = "Path to the input file",
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
    
    # Pack up
    return {
        'input_filepath_input'    : input_filepath_input   ,
        'output_folderpath_input' : output_folderpath_input,
    }