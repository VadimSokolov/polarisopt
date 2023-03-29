import subprocess
 
def run_sim_slurm(task_dir, polarisbin, scenariopath,convrgencepath,manager):
    d = manager.dictionary["slurm"]
    with open(f'./data/{d["scripttemplate"]}','r') as fh:   
        s = fh.read()
    s = s.replace("$JOBNAME", d["name"])
    s = s.replace("$NCPUS", d["ncpus"])
    s = s.replace("$MEM", d["mem"])
    s = s.replace("$OUTPUTFOLDER", task_dir)
    num_threads = d["ncpus"]
    cmd = f"cd {task_dir}\n"
    if convrgencepath is not None:
        control_fp=create_conv_files(scenariopath, polarisbin, convrgencepath, num_threads)
        cmd += " ".join(['python', os.path.join(convrgencepath,'run_convergence.py'),control_fp,task_dir])
    else:
        cmd += " ".join([polarisbin, scenariopath, num_threads])
    s = s.replace("$SCRIPT", cmd)
    manager.run_id+=1
    slurmfn = f'{task_dir}/{d["name"]}-{manager.run_id}.slurm'
    with open(slurmfn,'w') as fh:
        fh.write(s)
    result = subprocess.run(f"sbatch {slurmfn}", shell=True, capture_output=True, text=True)
    return print(f"Slurm task with {slurmfn} completed. Result: {result}")