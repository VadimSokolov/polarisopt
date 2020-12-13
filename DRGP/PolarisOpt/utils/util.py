"""
    This file contains the master calls to implement:
        * Build_Sampleset - A latin hypercube sample generator and evaluator
        * Calibrate_Simulation - the dimension-reduced Bayesian Optimization calibration procedure 
"""

import os, sys
@@ -9,10 +9,6 @@ import numpy as np
import threading
import time

def thread_it(f, a):
    r"""Generic Threading Function 
    Args:
        f (function): function to run in each thread
        a (list): list of argument instances within a () ex: [([10], 'save.txt'), ([11], 'save.txt')]
    Returns:
        joined threads
    """
    threads = []
    for item in a:
        t = threading.Thread(target = f, args = item)
        threads.append(t)
    
    for thread in threads:
        thread.start()
        time.sleep(20)
    
    for thread in threads:
        thread.join()
        #prevents us from doing anything until all are done


def convert_2str(inputs):
    if not isinstance(inputs,str):
        try:
            inputs = ' '.join(map(str,inputs))
        except:
            inputs = str(inputs)
    return inputs
