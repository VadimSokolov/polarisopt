#!/usr/bin/python
# Filename: run_convergence.py

import shutil
# from pathlib import Path
import sys
import os
import subprocess
from shutil import copyfile
from pathlib import Path
import json
import sqlite3
import csv
import traceback
import regression
import modify_scenario


# +++++++++++++++++++++++++++++++++++++++
# Run Polaris model for DTA convergence
# +++++++++++++++++++++++++++++++++++++++


def run_polaris_local(results_dir, exe_name, scenario_file, num_threads):
    # subprocess.call([exeName, arguments])
    out_file = open(str(results_dir / 'simulation_out.log'), 'w+')
    err_file = open(str(results_dir / 'simulation_err.log'), 'w+')
    proc = subprocess.Popen([str(exe_name), str(scenario_file), num_threads], stdout=out_file, stderr=subprocess.PIPE)
    for line in proc.stderr:
        sys.stdout.write(str(line))
        err_file.write(str(line))
    proc.wait()
    out_file.close()
    err_file.close()
    if proc.returncode != 0:
        sys.exit("POLARIS did not execute correctly")


def copyreplacefile(filename, dest_dir):
    dest_file = Path(dest_dir / filename.name)
    if dest_file.exists():
        os.remove(str(dest_file))
    copyfile(str(filename), str(dest_file))


def execute_sql_script(db_name, script):
    print('Executing Sqlite3 script: %s on database: %s' % (db_name, script))
    with open(script, 'r') as sql_file:
        sql_script = sql_file.read()

    db = sqlite3.connect(db_name)
    cursor = db.cursor()
    try:
        cursor.executescript(sql_script)
        db.commit()
    except sqlite3.Error as err:
        print('SQLite error: %s' % (' '.join(err.args)))
        print("Exception class is: ", err.__class__)
        print('SQLite traceback: ')
        exc_type, exc_value, exc_tb = sys.exc_info()
        print(traceback.format_exception(exc_type, exc_value, exc_tb))
    db.close()


def dump_table_to_csv(db, table, csv_name, loop):
    db_input = sqlite3.connect(db)
    sql3_cursor = db_input.cursor()
    query = 'SELECT * FROM ' + table
    sql3_cursor.execute(query)
    with open(csv_name, 'w') as out_csv_file:
        csv_out = csv.writer(out_csv_file, lineterminator='\n')  # gets rid of blank lines - defaults to \r\n
        if loop == 0:
            # write header
            csv_out.writerow([d[0] for d in sql3_cursor.description])
        # write data
        for result in sql3_cursor:
            csv_out.writerow(result)
    db_input.close()


def append_file(src, tar):
    with tar.open("a") as tar_file:  # append mode
        src_file = src.read_text()
        tar_file.write(src_file)
        tar_file.close()


def run_conv(control_file_name):
    control_file = Path(control_file_name)
    if not control_file.exists():
        print('ERROR: %s does not exist!' % control_file.name)
        return None  # or: raise

    with control_file.open() as f:
        try:
            json_data = json.load(f)
        except ValueError as e:
            print('invalid json: %s' % e)
            return None  # or: raise

    json_formatted_str = json.dumps(json_data, indent=2)

    print(json_formatted_str)

    # exe_name = sys.argv[1]
    # results_dir = sys.argv[2]

    # -- SET THE MAIN AND DTA RUN SCENARIO FILES
    # set python_path=C:\Program Files\Python36\
    # set sqlite3_path=%CD%\third_party\Sqlite3\
    # set tail_app=%CD%\third_party\baretail.exe
    # set cloud_backup_path=\\vms-fs2\VMS_FY19_SMART_Runs\Chicago_ABM_Convergence2018\Chicago_202000406_results\
    #
    # set SCENARIO_MAIN_INIT=scenario_abm_init_10per.json
    # set SCENARIO_MAIN=scenario_abm.json

    # python_path = json_data["python_path"]
    # sqlite3_path = json_data["sqlite3_path"]
    # tail_app = json_data["tail_app"]
    # cloud_backup_path = json_data["cloud_backup_path"]

    database_name = json_data["database_name"]

    scenario_main_init = json_data["scenario_main_init"]
    scenario_main = json_data["scenario_main"]

    exe_name = Path(json_data["model"])
    data_dir = Path(json_data["data"])
    results_dir = data_dir / json_data["results_dir"]
    # scenario_file = json_data["model_scenario"]
    num_threads = json_data["num_threads"]
    # standard_dir = json_data["standard_dir"]

    # -- ENTER THE NUMBER OF MAIN RUNS AND DTA RUNS PER MAIN RUN

    num_abm_runs = json_data["num_abm_runs"]

    # -- SET THE OUTPUT DIRECTORIES AS SPECIFIED IN THE SCENARIO FILES

    # output_directories = json_data["output_directories"]

    # set OUT_DIR_MAIN=linux_vs17_chicago2018_abm
    #
    # -- SET THE POLARIS RUN INFORMATION
    # set THREADS=20
    # set DB=chicago2018
    # set POLARIS_EXE=bin_new\Integrated_Model.exe

    # -------------------------------------------------------------------------------------------
    # Do not modify below here
    # -------------------------------------------------------------------------------------------

    # cwd = os.getcwd()
    # app_dir = Path.cwd()
    print(data_dir)
    os.chdir(str(data_dir))
    working_dir = Path.cwd()

    # del artificial_count.csv
    # del gap_calculations.csv
    # del gap_breakdown.csv
    if Path("artificial_count.csv").exists():
        os.remove("artificial_count.csv")
    if Path("gap_calculations.csv").exists():
        os.remove("gap_calculations.csv")
    if Path("gap_breakdown.csv").exists():
        os.remove("gap_breakdown.csv")

    # store the original inputs
    demand_db_name = database_name + "-Demand.sqlite"
    result_db_name = database_name + "-Result.sqlite"
    demand_db = working_dir / "backup" / demand_db_name
    skim_file_name = "highway_skim_file.bin"
    skim_file = working_dir / "backup" / skim_file_name
    output_file = results_dir / "simulation_out.log"

    copyreplacefile(demand_db, working_dir)
    copyreplacefile(skim_file, working_dir)
    print("Polaris output goes here: \'" + str(output_file) + "\'")

    # Process main ABM run

    # list of result directories
    result_paths = []

    for loop in range(0, int(num_abm_runs)):
        # Create results Directory if don't exist
        if not results_dir.exists():
            os.mkdir(str(results_dir))
            print("Directory ", results_dir, " Created ")
        else:
            print("Directory ", results_dir, " already exists")

        if loop == 0:
            scenario_file = scenario_main_init
        else:
            scenario_file = scenario_main
            modify_scenario.modify(scenario_main, "time_dependent_routing_weight_factor", 1.0)
            print("\nRunning Polaris SCENARIO_MAIN instance ", loop, " - see", results_dir / 'simulation_out.log')

        arguments = scenario_file + ' ' + num_threads
        print('Executing \'' + str(exe_name) + ' ' + arguments + '\'')

        run_polaris_local(results_dir, exe_name, scenario_file, num_threads)

        all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
        latest_subdir = Path(max(all_subdirs, key=os.path.getmtime))

        # standard_dir = 'Regression_test'
        result_paths.append(Path(latest_subdir))

        # move the output files (now that we know where the simulation files were created)
        results_dir_moved = working_dir / latest_subdir / json_data["results_dir"]
        print('Moving: ', results_dir, ' to:', results_dir_moved)
        shutil.move(str(results_dir), str(results_dir_moved))
        # os.rename('./simulation_out.log', simulated_dir + '/simulation_out.log')
        # os.rename('./simulation_err.log', simulated_dir + '/simulation_err.log')

        # copy local results back to the main run directory for the next run
        copyreplacefile(working_dir / latest_subdir / demand_db_name, working_dir)
        copyreplacefile(working_dir / latest_subdir / result_db_name, working_dir)
        copyreplacefile(working_dir / latest_subdir / skim_file_name, working_dir)

        # %sqlite3_path%sqlite3.exe "%WORKDIR%\%DB%-Demand.sqlite" < clean_db_after_abm_for_abm.sql
        execute_sql_script(working_dir / demand_db_name, working_dir / "clean_db_after_abm_for_abm.sql")

        # ren %WORKDIR%\!out_local!\summary.csv summary_abm_%%S.csv
        latest_demand_db = working_dir / latest_subdir / demand_db_name
        execute_sql_script(latest_demand_db, working_dir / "wtf_baseline_analysis_25Per_calibrate.sql")
        execute_sql_script(latest_demand_db, working_dir / "gap_calculate.sql")
        # execute_sql_script(working_dir / latest_subdir / demand_db_name, working_dir / "output_to_csv.sql")
        dump_table_to_csv(latest_demand_db, "artificial_count", working_dir / "artificial_count_temp.csv", loop)
        dump_table_to_csv(latest_demand_db, "gap_calculations", working_dir / "gap_calculations_temp.csv", loop)
        dump_table_to_csv(latest_demand_db, "gap_breakdown", working_dir / "gap_breakdown_temp.csv", loop)

        # append temp data to main file
        append_file(working_dir / "artificial_count_temp.csv", working_dir / "artificial_count.csv")
        append_file(working_dir / "gap_calculations_temp.csv", working_dir / "gap_calculations.csv")
        append_file(working_dir / "gap_breakdown_temp.csv", working_dir / "gap_breakdown.csv")

        os.remove(working_dir / "artificial_count_temp.csv")
        os.remove(working_dir / "gap_calculations_temp.csv")
        os.remove(working_dir / "gap_breakdown_temp.csv")

        if loop > 0:
            print('Checking convergence on \'' + str(latest_subdir) + '\'')
            regression.regression(result_paths[loop - 1].name, result_paths[loop].name)

        # make sure results_dir exists
        # if not os.path.exists(results_dir):
        #    os.makedirs(results_dir)

        # copyfile('./Regression_Report.html', results_dir + '/Regression_Report.html')
        # copyfile(simulated_dir + '/in_network.png', results_dir + '/in_network.png')
        # copyfile(simulated_dir + '/simulation_out.log', results_dir + '/simulation_out.log')
        # copyfile(simulated_dir + '/simulation_err.log', results_dir + '/simulation_err.log')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage %s <json_control_file>' % (sys.argv[0]))
        sys.exit(-1)

    run_conv(sys.argv[1])
