import json


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