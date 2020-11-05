import os
import sys
from  subprocess import Popen
import shutil
import numpy as np
from PolarisOpt.utils import archiver
import sqlite3
import json
from PolarisOpt.utils.objective_funcs import run_objective

def run_task(problem_info,inputs,task):
   r"""Evaluates a set of inputs in the simulator

   Args:
      problem_info (class): object containing settings for simulation
      inputs (n-array): the new values to be run
      task (int): counter indicating simulation instance being performed
        
   Returns:
      the single-output objective function and uncollapsed distance from target based on the outputs
   """
   task_dir=os.path.join(problem_info.working_dir,'experiments',"Sim"+str(task))
   src_dir=problem_info.simulation_path  
   if os.path.exists(task_dir):
      shutil.rmtree(task_dir)
   shutil.copytree(src_dir,task_dir)

   archiver.update_json(problem_info.vnames,inputs,task_dir)
   
   exe_fn = os.path.join(problem_info.simulation_path,'Integrated_model.exe')
   run_sim(task_dir, exe_fn, problem_info.simulation_scenario_name)

   return pull_result(task_dir, problem_info)

def run_sim(task_dir, exe_fn, sc_fn):
   r"""runs the POLARIS executable with the associated scenario file
   Args:
      task_dir: (path) path to the folder containing the scenario and variable-related files
      exe_fn: (path) full path to the Polaris Integrated_model.exe
      sc_fn: (path) scenario filename within task_dir for the .json needed to run an instance of the executable
   """
   cur_dir=os.getcwd()
   os.chdir(task_dir)
   # Run the Polaris exe file via pipe
   p1 = Popen([exe_fn, sc_fn], shell=True)
   p1.wait()
   os.chdir(cur_dir)
   return print("task completed")


def pull_result(task_dir, problem_info):
   r"""Performs a SQL query to retrieve the results and calculates the objective value
   Args:
      task_dir (path): folder path containing the sample-instance files
      problem_info (class): object containing settings for simulation
         simulation_path: path to the folder containing the simulation         
   Returns:
      the uncollapsed distance from the target outputs and objective
      value
   """
   sc_fn = os.path.join(task_dir,problem_info.simulation_scenario_name)
   dictionary = json.loads(open(sc_fn).read())
   output_dir = dictionary['Output controls']['output_dir_name']
   task_db = os.path.join(
      task_dir,
      output_dir,
      os.path.split(problem_info.target_output_filename)[1]
      )
   target_db = problem_info.target_output_filename
  
   new1=sqlite3.connect(task_db,timeout=60)
   ref1=sqlite3.connect(target_db,timeout=60)
   cur1= new1.cursor()
   cur2= ref1.cursor()
   cur1.execute(problem_info.output_SQL_query)
   new_output = cur1.fetchall()
   cur2.execute(problem_info.output_SQL_query)
   ref_output = cur2.fetchall()
   new1.close()
   ref1.close()
   
   new_output=np.asarray(new_output)
   ref_output=np.asarray(ref_output)  
   if new_output.shape[0] != ref_output.shape[0]:
      print('output mismatch by % and %' % (new_output.shape,ref_output.shape))
      obj="P"
      er=None
      return obj,er
   else:
      return run_objective(ref_output[:,1]-new_output[:,1],problem_info.objective_type)


def eval_sample_task(problem_info,output_fn,inputs,task):
   r"""Evaluates a set of inputs generated in the original subspace and records the outcome

   Args:
      problem_info (class): object containing settings for simulation
         simulation_path: path to the folder containing the simulation
      output_fn (path): full path to file outputs should be saved to
      inputs (n-array): the new values to be run
      task (int): counter indicating simulation instance being performed
        
   Returns:
      a results file containing the [uncollapsed objective outputs, inputs]
   """
   print("running sample %d \n" % task)   

   _, y_diff = run_task(problem_info,inputs,task)
   archiver.replace_pend(inputs,y_diff,output_fn)


def eval_DR_task(problem_info,DR_model,DR_input,task):
   r"""Evaluates a set of inputs and records the outcome

   Args:
      problem_info (class): object containing settings for simulation
         simulation_path: path to the folder containing the simulation 
      DR_model (class): the model using the Dimension Reduction
      DR_inputs (n-array): the new values for 
      task (int): counter indicating simulation instance being performed
        
   Returns:
      a results file containing the [collapsed objective function, DR_inputs]
      an 'orig' results file containing the [uncollapsed objective function, detransformed inputs]
   """
   print("running sample %d \n" % task)   

   xhat = DR_model.decode_X(DR_input)
   y_single, y_diff = run_task(problem_info,xhat,task)

   archiver.replace_pend(DR_input,y_single,problem_info.res_filename)
   archiver.record_eval(xhat,y_diff,problem_info.reso_filename)

