import sys
import json

def read_general():
    print("\nReading Input")

    # Grab command line input
    example_name = sys.argv[1]

    # Setting reader booleans
    use_json_reader = False

    # Check filename extension
    if example_name.endswith('.json'):
        filename = example_name
        use_json_reader = True
    else:
        filename = f"{example_name}.json"
        use_json_reader = True
    
    if use_json_reader:
        input_parameters = read_json(filename)

    return {
        'input_filename': filename,
        'input_parameters': input_parameters,
    }

def read_json(filename):
    
    with open(filename, "r") as file:
       parameters_input = json.load(file)
    print(f"  Successfully read input: {filename}")

    return parameters_input