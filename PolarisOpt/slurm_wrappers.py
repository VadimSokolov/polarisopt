import subprocess
import os
# from PolarisOpt.eval_sim import create_conv_files

def run_sim_slurm(task, manager):
    scenariopath = os.path.join(task.task_dir, manager.simulation_scenario_name)
    d = manager.dictionary["slurm"]
    with open(os.path.join(manager.working_dir,d["scripttemplate"]),'r') as fh:   
        s = fh.read()
    jobname = d["name"]
    num_threads = d["ncpus"]
    s = s.replace("$JOBNAME", f"{jobname}-{task.run_id}")
    s = s.replace("$NCPUS", num_threads)
    s = s.replace("$MEM", d["mem"])
    s = s.replace("$OUTPUTFOLDER", task.task_dir)
    
    cmd = f"cd {task.task_dir}\n"
    cmd += f'cp -r {os.path.dirname(manager.polaris_executable)} bin\n'
    polarisbin = f'./bin/{os.path.basename(manager.polaris_executable)}'
    # if convrgencepath is not None:
    #     control_fp=create_conv_files(scenariopath, polarisbin, convrgencepath, num_threads)
    #     cmd += " ".join(['python', os.path.join(convrgencepath,'run_convergence.py'),control_fp,task_dir.task_dir])
    # else:
    #     cmd += " ".join([polarisbin, scenariopath, num_threads])
    if manager.convergence:
        with open(os.path.join(manager.working_dir,manager.convergence_path),'r') as fh:
            pyscript = fh.read()
        pyscript = pyscript.replace("$POLARISBIN", "'"+polarisbin+"'")
        pyscript = pyscript.replace("$PRJDIR", "'"+task.task_dir+"'")
        pyscript = pyscript.replace("$DBNAME", "'"+jobname+"'")
        pyscript = pyscript.replace("$NCPUS", num_threads)
        pyscript = pyscript.replace("$NRUNS", str(manager.num_abm_runs))
        pyscript = pyscript.replace("$RESTART", str(task.start_iteration_from)) 
        convfn  = f'{task.task_dir}/{d["name"]}-{task.run_id}.py'
        with open(convfn,'w') as fh:
            fh.write(pyscript)
        cmd += " ".join(['python', convfn])
    else:
        cmd += " ".join([polarisbin, scenariopath, num_threads])
    s = s.replace("$SCRIPT", cmd)
    slurmfn = f'{task.task_dir}/{d["name"]}-{task.run_id}.slurm'
    with open(slurmfn,'w') as fh:
        fh.write(s)
    print (f"Submitting slurm task with {slurmfn}")
    result = subprocess.run(f"sbatch {slurmfn}", shell=True, capture_output=True, text=True)
    if result.returncode!=0:
        print(f"\nSlurm task with {slurmfn} failed.\nResult: {result}\n")
        print(result.stderr)
        return False
    # Check if finieshed file was created
    task_output = manager.get_task_output(task.task_dir,scenariopath)
    if manager.convergence:
        lastit, unfinishedit = manager.check_iterations(task.task_dir,scenariopath)
        if lastit < manager.num_abm_runs:
            print(f"Last iteration for {task_output} was {lastit} which is below required {manager.num_abm_runs}. The unfinished iterations are {unfinishedit}")
            return False
    else:        
        if not os.path.exists(os.path.join(task_output,'finished')):
            print(f"Finished file was not created in {task_output}")
            return False 
    print(f"\nSlurm task with {slurmfn} completed.\nResult: {result}\n")
    return True