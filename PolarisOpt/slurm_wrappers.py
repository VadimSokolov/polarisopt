from PolarisOpt import eval_sim

def run_sim_slurm(task_dir, exe_fp, sc_fp,cv_fp,manager):
    d = manager.dictionary["slurm"]
    with open(d["scripttemplate"],'r') as fh:   
        s = fh.read()
    s = s.replace("$NAME", d["name"])
    s = s.replace("$NCPUS", d["ncpus"])
    s = s.replace("$NAMMEME", d["mem"])
    cmd = f"cd {task_dir}\n"
    if cv_fp is not None:
        control_fp=create_conv_files(sc_fp, exe_fp, cv_fp, num_threads)
        cmd += " ".join(['python', os.path.join(cv_fp,'run_convergence.py'),control_fp,task_dir])
    else:
        cmd += " ".join([exe_fp, sc_fp, num_threads])
    s = s.replace("$SCRIPT", cmd)
    manager.run_id+=1
    slurmfn = f'{d["name"]}-{manager.run_id}.slurm'
    with open(slurmfn,'w') as fh:
        fh.write(s)
    result = subprocess.run(f"sbatch {slurmfn}", shell=True, capture_output=True, text=True)
    return print("task completed")