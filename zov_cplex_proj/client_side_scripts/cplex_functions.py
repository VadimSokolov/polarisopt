from pathlib import Path
import datetime
from dateutil.tz import tzlocal
import subprocess
import shlex
import os
import glob

import json
import socket
import sys
import os
from os import walk
import shutil
import bebop_parsl

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def time(format="%Y%m%d_%H%M%S"):
    return datetime.datetime.now(tzlocal()).strftime(format)
    
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
    print("Source Directory: {}".format(str(source_directory)))
    print("Input Directory: {}".format(str(input_directory)))
    print("Results Directory: {}".format(str(results_directory)))

    # Runs secondary scripts. Skip bebop calls for local testing
    concat_inputs(source_directory, input_directory)

    # time python bebop_preprocess.py $timestamp
    results_file = bebop_parsl.submit_cplex_preprocess(timestamp)

    # time python bebop_submit.py $input_directory $timestamp
    for fname in os.listdir(input_directory):
        if fname.endswith('.txt'):
            # do stuff on the file
            break
        else:
            print('Input files do not exist in {}'.format(input_directory))
            sys.exit(1)

    control_fname = "polaris.json"
    with open(control_fname) as control_file:
        json_data = json.load(control_file)
        
    bebop_input = json_data["input_directory"]
    bebop_results = json_data["results_directory"]

    bebop_parsl.submit_cplex_jobs(str(input_directory), bebop_results, timestamp)

    # time python bebop_postprocess.py $results_directory $timestamp
    if not os.path.exists(results_directory):
        os.mkdir(results_directory)

    results_file = bebop_parsl.submit_cplex_postprocess(str(results_directory), timestamp)
    #print("Copied results to '{}'".format(results_directory))
    
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
            #print("no more files to match")
            break
           
def send_email(results_directory):
    you = []
    email_list = './email_list.txt'
    if os.path.isfile(email_list) is not True :
        print('The email recipient list file \'{}\' does not exist').format(email_list)
        return
                                
    with open(email_list) as fh:
        you = fh.read().splitlines()
        if (len(you) < 1) :
            print('There are no email recipients specified')
            return
                                                                                
            
    print('Email will be sent to:')
    print(you)
    me = "polaris_testing@anl.gov"
    #you = 'rweimer@anl.gov'
    msg = MIMEMultipart()
    msg['Subject'] = 'Bebop job completed'
    msg['From'] = me
    msg['To'] = ", ".join(you)
    msg.preamble = 'Bebop has completed a job'
    content = 'The results can be found in folder:\n\n%s'%str(results_directory)
    text_part = MIMEText(content, 'plain')
    msg.attach(text_part)
    s = smtplib.SMTP('mailhost.anl.gov', 25)
    s.sendmail(me, you, msg.as_string())
    s.quit()

