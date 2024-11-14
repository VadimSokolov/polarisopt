from PolarisOpt import sampler, manager
import numpy as np

settings_filepath = '/home/vsokolov/projects/timing-austin/latin-study/settings_slurm.json'
s = sampler.MorrisSampler('/home/vsokolov/projects/timing-austin/latin-study/config_morris_timing.json','/home/vsokolov/projects/timing-austin/latin-study/latin-study',1,2)
m = manager.Manager(settings_filepath,s)
m.run_study()