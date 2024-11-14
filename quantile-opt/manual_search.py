from PolarisOpt import sampler, manager
import numpy as np

settings_filepath = '/home/vsokolov/projects/timing-austin/quantile-opt/settings_slurm.json'
default = [-0.1,0,0,-0.1,-0.1,-0.1,-0.31,-0.71,-0.32,0,0,0]
manual = [-0.3234, -1.3213,  0.7050, -1.8785,  0.6472,  0.2259, -1.4005, -0.3865,-0.4250, -3.4885, -0.2371, -0.9708]
varnames=['S_AMPEAK_TT','S_AMOFFPEAK_TTV','S_AMOFFPEAK_TT','S_PMOFFPEAK_TT','S_PMPEAK_TT','S_EVENING_TT','S_AMOFFPEAK_OCCUPANCY','S_PMOFFPEAK_OCCUPANCY','S_PMPEAK_OCCUPANCY','D_AMPEAK_OCCUPANCY','D_AMOFFPEAK_OCCUPANCY','D_PMOFFPEAK_OCCUPANCY']
varfiles = ['CampoTimingChoiceModel.json']*12
X = np.array([default,manual])
s = sampler.ManualSampler(X,varnames,varfiles,'quantile-samples')
m = manager.Manager(settings_filepath,s)
m.run_study()