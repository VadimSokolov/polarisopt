import io;
import sys;
import files;
import R;
import python;
import string;

string emews_root = getenv("EMEWS_PROJECT_ROOT");
string turbine_output = getenv("TURBINE_OUTPUT");
string scenario_dir = argv("scenario_dir");

json_param_code =
"""
import json
import os
import ast

json_input = '%s'  # arg 1
instance = '%s'  # arg 2
line = '%s'  # arg 3

with open(json_input) as f_in:
	params = json.load(f_in)

new_params = ast.literal_eval(line)
params.update(new_params)

params['output_dir_name'] =  os.path.join(instance, 'model_output')

path_to_new_json = os.path.join(instance,'scenario_init.json')  

with open(path_to_new_json, 'w') as f_out:
	json.dump(params, f_out)
""";

app (file out, file err) run_model (file shfile, string param_line, string instance, string num_threads)
{
    "bash" shfile param_line emews_root instance scenario_dir num_threads @stdout=out @stderr=err;
}

// call this to create any required directories
app (void o) make_dir(string dirname) {
  "mkdir" "-p" dirname;
}

string json_master = emews_root + "/data/polaris-data/" + scenario_dir + "/scenario_init.json";

file model_sh = input(emews_root+"/scripts/polaris.sh");
file upf = input(argv("f"));
string upf_lines[] = file_lines(upf);
foreach s,i in upf_lines {
  string instance = "%s/instance_%i/" % (turbine_output, i+1);
  make_dir(instance) => {
    file out <instance+"out.txt">;
    file err <instance+"err.txt">;

    # TODO read num threads from UPF line
    string num_threads="10";

    code = json_param_code % (json_master, instance, s);

    python_persist(code, "'ignore'") =>
    (out,err) = run_model(model_sh, s, instance, num_threads);
  }
}
