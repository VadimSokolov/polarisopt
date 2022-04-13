import sys
import os
import json
import shutil
import argparse

from PolarisOpt.setup_manager import SetupManager
from PolarisOpt import F
from PolarisOpt.utils.archiver import load_model
from PolarisOpt.F import calibrate_simulation
import proxies
import eq


def create_manager(params):
    emews_root = os.environ['EMEWS_PROJECT_ROOT']
    turbine_output = os.environ['TURBINE_OUTPUT']
    params['emews_root'] = emews_root
    data_dir = os.path.join(emews_root, 'data')
    settings_file = os.path.join(data_dir, params['settings_file'])
    config_file = os.path.join(data_dir, params['config_file'])

    settings_file_dst = os.path.join(turbine_output, os.path.basename(settings_file))
    config_file_dst = os.path.join(turbine_output, os.path.basename(config_file))

    shutil.copy(settings_file, settings_file_dst)
    shutil.copy(config_file, config_file_dst)

    with open(settings_file_dst) as f_in:
        settings = json.load(f_in)
        training_file = settings['File controls']['training_filename']
        training_file_dst = os.path.join(turbine_output, os.path.basename(training_file))
        res_file = settings['File controls']['res_filename']
        res_file_dst = os.path.join(turbine_output, os.path.basename(res_file))

        settings['File controls']['training_filename'] = training_file_dst
        settings['File controls']['res_filename'] = res_file_dst

        if os.path.exists(training_file):
            shutil.copy(training_file, training_file_dst)
        if os.path.exists(res_file):
            shutil.copy(res_file, res_file_dst)

    with open(settings_file_dst, 'w') as f_out:
        json.dump(settings, f_out, indent=4)

    manager = SetupManager(settings_file_dst, config_file_dst)
    return manager


def run_sampleset(params):
    num_samples = int(params['num_samples'])
    manager = create_manager(params)
    F.build_sampleset(manager, manager.training_filename, num_samples=num_samples, use_emews=True)


def run_calibration(params):
    manager = create_manager(params)
    data_dir = os.path.join(params['emews_root'], 'data')
    dr_model_file = os.path.join(data_dir, params['dr_model_file'])
    dr_model = load_model(dr_model_file)
    if params['m_model_file'] == '':
        m_model = None
    else:
        m_model_file = os.path.join(data_dir, params['m_model_file'])
        m_model = load_model(m_model_file)
    calibrate_simulation(manager, dr_model, m_model, quiet=params['quiet'], use_emews=True)


def create_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_id', help="experiment id")
    parser.add_argument('experiment_dir', help="experiment directory")
    parser.add_argument("config_file", help="configuration file (json format)")
    return parser

# def run():
#     """"""
#     os.chdir(os.environ['TURBINE_OUTPUT'])
#     eqpy.OUT_put("Params")
#     algo_params_file = eqpy.IN_get()
#     with open(algo_params_file) as f_in:
#         params = json.load(f_in)
#     print(params, flush=True)

#     run_type = params['run_type']
#     if run_type == 'sampleset':
#         run_sampleset(params)
#     elif run_type == 'calibration':
#         run_calibration(params)

#     eqpy.OUT_put("DONE")
#     eqpy.OUT_put("See DRGP Output for Results")


if __name__ == "__main__":
    parser = create_args_parser()
    args = parser.parse_args()
    os.environ['EXP_ID'] = args.experiment_id
    with open(args.config_file) as f_in:
        params = json.load(f_in)
    os.environ['TURBINE_OUTPUT'] = args.experiment_dir
    os.chdir(args.experiment_dir)
    # TODO create proxystore in experiment dir??
    proxies.init(args.experiment_id)
    # TODO init the database
    eq.init()
    run_type = params['run_type']
    if run_type == 'sampleset':
        run_sampleset(params)
    elif run_type == 'calibration':
        run_calibration(params)
