from abc import ABC, abstractmethod
import json
import numpy as np
import pandas as pd
import pickle
import os.path


# Obsolete, use inputdf instead
class VarConfig():
    def __intit__(self,config_filepath):
        variables = json.loads(open(config_filepath).read())   ##reads config file
    #    totaldim = np.sum([vkey["size"] for key in variables for vkey in variables[key]])
        self.vnames = [[key, [vkey["name"] for vkey in variables[key] for n in range(0, int(vkey["size"]))]] for key in variables]
        self.lower = [float(vkey["min"]) for key in variables for vkey in variables[key] for n in range(0, int(vkey["size"]))]
        self.upper = [float(vkey["max"]) for key in variables for vkey in variables[key] for n in range(0, int(vkey["size"]))]
        self.type = [vkey["type"] for key in variables for vkey in variables[key] for n in range(0, int(vkey["size"]))]
        self.flatvnames = [name for v in self.vnames for name in v[1]]
        self.dim = len(flatvnames)
def inputdf(config_filepath):
    with open(config_filepath,'r') as fh:
        d = json.load(fh)
    dflat = []
    for file in d:
        cnt = {'file':file}
        for r in d[file]:
            # dflat.append(cnt | r)
            # dflat.append(cnt.update(r))
            dflat.append({**cnt, **r})
    df = pd.DataFrame(dflat) # each row is a varibale
    return(df)

class Sample:
    def __init__(self, input, index = None):
        self.input = input # np array
        self.hash = hash(tuple(input)) # will use as saple's unique identifier 
        self.index = index # index of the sample in the original sample set
        # to be filled in be manager
        self.status = 'pending' # 'running', 'finished', 'failed'
        self.output = None
        self.folder = None


class Sampler(ABC):
    """
    This class manages samples and resutls of evaluating those sampels
    """
    def __init__(self,varnames,varfiles,samplefile):
        """
        Parameters
        ----------
        varnames : list of str
            names of the variables defined in control json files to be used as inputs 
        varfiles : list of str
            names of the corresponding json control files
        samplefile : str
            name of the file to wich the results will be stored (pickled and csv)
        """
        assert len(varnames)==len(varfiles)
        self.p = len(varnames) # number of input variables
        self.samplefile = samplefile # where resiults will be stored for futher analysis or restarting capabilities
        if os.path.isfile(samplefile):
            print('Loading samples from %s'%samplefile)
            with open(samplefile, 'rb') as fh:
                self.samples = pickle.load(fh)
        else:
            self.samples = [] # list of all the samples
        self.varnames = varnames
        self.varfiles = varfiles
        self.samplefile = samplefile
        self.namesfiles = dict(zip(varnames,varfiles))
    def topkl(self):
        """Dump samples to pickle"""
        with open(self.samplefile+'.pkl', 'wb') as fh:
            pickle.dump(self.samples,fh)
    def tocsv(self):
        """Dump samples into csv file"""
        fn = self.samplefile+'.csv'
        rows = []
        for item in self.samples:
            rows.append(item.input + [item.hash,item.status,item.folder,item.output])
        df = pd.DataFrame(rows, columns=self.varnames + ['hash','status','folder','output'])
        df.to_csv(fn)
    def getsinglesample(self,hash):
        s = list(filter(lambda x: x.hash==hash, self.samples))
        assert len(s)<2 # only can have one or none samples with this hash
        if len(s)==0:
            return None
        else:
            return s[0]
    def setstatus(self,hash, status):
        s = getsinglesample(hash)
        assert s is not None
        s.status = status
    def addsample(self,input):
        assert len(input) == self.p #make sure that we add the required number of values
        indices = [s.index for s in self.samples]
        if len(indices)==0:
            index = 0
        else:
            index = max(indices)+1
        s = Sample(input,index)
        test = self.getsinglesample(s.hash)
        if test is None: # avoid duplicates
            self.samples.append(s)
        else: #print a Warning
            print(f'WARNING: addsample funciton. Sample {input} was already added to the list of samples by')
        self.topkl()

    def getsamples(self, max = 9999999):
        # get list of unevaluated samples
        candidates = list(filter(lambda x: x.status=='pending', self.samples))[:max]
        # for s in candidates:
        #     s.status = status
        return candidates
    # @abstractmethod
    # def setoutputs(self,samples):
    #     """Takes the evaluated samples and potentially updates the samples list"""
    #     pass

class MorrisSampler(Sampler):
    def __init__(self,config_filepath,samplefile,N=8,num_levels=4, sampler='morris'):
        vardf = inputdf(config_filepath)
        varnames = list(vardf['name'].values)
        varfiles = vardf['file'].values
        super().__init__(varnames, varfiles, samplefile)
        if len(self.samples)==0:
            from SALib.sample import morris, latin
            problem = {'num_vars': vardf.shape[0],'names': varnames,'bounds': np.c_[vardf['min'].values, vardf['max'].values]}
            if sampler=='morris':
                print('Generating Morris samples')
                X = morris.sample(problem,N=N,num_levels=num_levels)
            if sampler=='latin':
                print('Generating Latin samples')
                X = latin.sample(problem=problem,N=N)
            for x in X:
                self.addsample(x)
    def setoutputs(self,samples):
        pass # do nothing, Morris procedure does not assume iterative design

class ManualSampler(Sampler):
    def __init__(self,X,varnames, varfiles, samplefile):
        assert X.shape[1]==len(varnames)
        super().__init__(varnames, varfiles, samplefile)
        for x in X:
            self.addsample(x)




