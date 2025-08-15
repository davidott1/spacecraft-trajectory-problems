from src.load.parser     import parse_command_line_interface_input
from src.load.reader     import read_input_parameters
from src.load.configurer import configure_validate_input, configure_output


def optimal_trajectory_input():

    print("\nINPUT PROCESS")
    files_folders_params_input = parse_read_configure()
    files_folders_params       = configure_validate_input(files_folders_params_input)

    return files_folders_params


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

