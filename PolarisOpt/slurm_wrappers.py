import subprocess
import os
# from PolarisOpt.eval_sim import create_conv_files

def run_sim_slurm(task_dir, polarisbin, scenariopath,convrgencepath,manager,run_id):
    d = manager.dictionary["slurm"]
    with open(os.path.join(manager.working_dir,d["scripttemplate"]),'r') as fh:   
        s = fh.read()
    jobname = d["name"]
    s = s.replace("$JOBNAME", f"{jobname}-{run_id}")
    s = s.replace("$NCPUS", d["ncpus"])
    s = s.replace("$MEM", d["mem"])
    s = s.replace("$OUTPUTFOLDER", task_dir)
    num_threads = d["ncpus"]
    cmd = f"cd {task_dir}\n"
    # if convrgencepath is not None:
    #     control_fp=create_conv_files(scenariopath, polarisbin, convrgencepath, num_threads)
    #     cmd += " ".join(['python', os.path.join(convrgencepath,'run_convergence.py'),control_fp,task_dir])
    # else:
    #     cmd += " ".join([polarisbin, scenariopath, num_threads])
    cmd += " ".join([polarisbin, scenariopath, num_threads])
    s = s.replace("$SCRIPT", cmd)
    slurmfn = f'{task_dir}/{d["name"]}-{run_id}.slurm'
    with open(slurmfn,'w') as fh:
        fh.write(s)
    print (f"Submitting slurm task with {slurmfn}")
    result = subprocess.run(f"sbatch {slurmfn}", shell=True, capture_output=True, text=True)
    if result.returncode!=0:
        print(f"\nSlurm task with {slurmfn} failed.\nResult: {result}\n")
        print(result.stderr)
        return False
    task_output = manager.get_task_output(task_dir,scenariopath)
    if not os.path.exists(os.path.join(task_output,'finished')):
        print(f"Finished file was not created in {task_output}")
        return False 
    return f"\nSlurm task with {slurmfn} completed.\nResult: {result}\n"