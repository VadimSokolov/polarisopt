from pathlib import Path
import datetime
import subprocess
import shlex
import os
import glob

def time(format="%Y%m%d%H%M%S"):
    return datetime.datetime.utcnow().strftime(format)
    
def make_directories(base_directory, timestamp)->Path:
    source_directory = Path(base_directory) / f"ZOV"
    
    # directory should be ex. "run_2019103102731"
    folder_name = "run_" + timestamp
    run_directory = Path(base_directory) / folder_name
    run_directory.mkdir(exist_ok=True)
    
    input_directory =Path(run_directory) / f"inputs"
    input_directory.mkdir(exist_ok=True)
    
    results_directory=Path(run_directory) / f"results"
    results_directory.mkdir(exist_ok=True)
    
    # return as a groups, python nonsense
    return source_directory, run_directory, input_directory, results_directory
    
def run_external_scripts(timestamp, source_directory, input_directory, results_directory):    
    print(str(source_directory))
    print(str(input_directory))

    # Runs secondary scripts. Skip bebop calls for local testing
    concat_inputs(source_directory, input_directory)

    # time python bebop_preprocess.py $timestamp

    # time python bebop_submit.py $input_directory $timestamp

    # time python bebop_postprocess.py $results_directory $timestamp
    
def concat_inputs(input_dir, output_dir):
    firstTen = open(output_dir/f"all_0.txt", 'w')
    for f in Path(input_dir).glob('*_[0-9].txt'):
        file = open(f, 'r')
        firstTen.write(file.read())
    for f in Path(input_dir).glob('*_[0-9][0-9].txt'):
        file = open(f, 'r')
        firstTen.write(file.read())
    firstTen.close()
    
    for i in range(1, 21):
        # lots of steps to test if we're out of files
        path = f'*_' + str(i) + '[0-9][0-9].txt'
        path = input_dir/path
        g = glob.glob(str(path))        
        if (g):
            filePath = 'all_' + str(i) + '.txt'
            currFile = open(output_dir/filePath, 'w')
            for f in g:
                file = open(f, 'r')
                currFile.write(file.read())
            currFile.close();
        else:
            print("no more files to match")
            break
            