import json
import os

def create_scenario_init(json_input, instance):

with open(json_input) as f_in:
	params = json.load(f_in)

params['output_dir_name'] =  os.path.join(instance, 'model_output')

path_to_new_json = os.path.join(instance,'scenario_init.json')  

with open(path_to_new_json, 'w') as f_out:
	json.dump(params, f_out)



