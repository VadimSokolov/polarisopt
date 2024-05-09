import subprocess
import os
# from PolarisOpt.eval_sim import create_conv_files

def run_sim_slurm(task, polarisbin, scenariopath,manager):
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
    task_output = manager.get_task_output(task.task_dir,scenariopath)
    if not os.path.exists(os.path.join(task_output,'finished')):
        print(f"Finished file was not created in {task_output}")
        return False 
    print(f"\nSlurm task with {slurmfn} completed.\nResult: {result}\n")
    return True