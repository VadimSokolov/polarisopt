from pathlib import Path
from polaris.runs.convergence.convergence_iteration import ConvergenceIteration
from polaris.runs.wtf_runner import run_baseline_analysis
from concurrent import futures
import shutil


project_dir=Path("/home/vsokolov/projects/timing-austin/quantile-opt/experiments")
pattern = 'Sim*/Austin/'
dirs = [d.parent for d in project_dir.glob(f'{pattern}/finished')]

# def bl(d):
#     print('Starting baseline analysis for', d)
#     shutil.copy(d.parent/'Austin-Supply.sqlite',d/'Austin-Supply.sqlite')
#     it = ConvergenceIteration.from_dir(dir=d, db_name='Austin', it_num=1,it_type='skim')
#     run_baseline_analysis(it,0.25)
#     print('Finished baseline analysis for', d)
#     return None
# with futures.ThreadPoolExecutor(40) as executor:
#     result = executor.map(bl, dirs)


from polaris.runs.calibrate import timing_choice
from polaris.runs.polaris_inputs import PolarisInputs
from polaris.utils.dict_utils import denest_dict
import pandas as pd
target = timing_choice.load_target("/home/vsokolov/models/Austin/calibration_targets/timing_choice_targets.csv");
dc = denest_dict(target)
dc['sim'] = -1
d = pd.DataFrame([dc])

def get_timing_choice(dd):
    output_dbs = PolarisInputs.from_dir(dd, db_name='Austin')
    simulated = timing_choice.load_simulated(output_dbs, 0.25, False)
    dc = denest_dict(simulated)
    dc['sim'] = dd
    new_df = pd.DataFrame([dc])
    return(new_df)

with futures.ThreadPoolExecutor(40) as executor:
    result = executor.map(get_timing_choice, dirs)
    for new_df  in result:
        d = pd.concat([d, new_df], ignore_index=True)

d.to_csv('quantile-opt/timing_choice_distribution.csv')