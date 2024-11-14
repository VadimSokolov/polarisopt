import subprocess
import os
# from PolarisOpt.eval_sim import create_conv_files

def run_sim_slurm(sample, manager):
    scenariopath = os.path.join(sample.folder, manager.polaris_scenario_file)
    d = manager.dictionary["slurm"]
    with open(os.path.join(manager.working_dir,d["scripttemplate"]),'r') as fh:   
        s = fh.read()
    jobname = d["name"]
    num_threads = d["ncpus"]
    s = s.replace("$JOBNAME", f"{jobname}-{sample.index}")
    s = s.replace("$NCPUS", num_threads)
    s = s.replace("$MEM", d["mem"])
    s = s.replace("$OUTPUTFOLDER", sample.folder)
    
    cmd = f"cd {sample.folder}\n"
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
        pyscript = pyscript.replace("$PRJDIR", "'"+sample.folder+"'")
        pyscript = pyscript.replace("$DBNAME", "'"+jobname+"'")
        pyscript = pyscript.replace("$NCPUS", num_threads)
        pyscript = pyscript.replace("$NRUNS", str(manager.num_abm_runs))
        pyscript = pyscript.replace("$RESTART", str(sample.start_iteration_from)) 
        convfn  = f'{sample.folder}/{d["name"]}-{sample.index}.py'
        with open(convfn,'w') as fh:
            fh.write(pyscript)
        cmd += " ".join(['python', convfn])
    else:
        cmd += " ".join([polarisbin, scenariopath, num_threads])
    s = s.replace("$SCRIPT", cmd)
    slurmfn = f'{sample.folder}/{d["name"]}-{sample.index}.slurm'
    with open(slurmfn,'w') as fh:
        fh.write(s)
    print (f"Submitting slurm task with {slurmfn}")
    sample.status = 'running'
    result = subprocess.run(f"sbatch {slurmfn}", shell=True, capture_output=True, text=True)
    if result.returncode!=0:
        print(f"\nSlurm task with {slurmfn} failed.\nResult: {result}\n")
        print(result.stderr)
        return False
    return sample
