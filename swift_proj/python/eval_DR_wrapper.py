import proxies

from PolarisOpt import eval_sim


@proxies.app
def run(manager, dr_model, pend_samples, row: int):
    pend_sample = pend_samples[row]
    eval_sim.eval_DR_task(manager, dr_model, pend_sample, row)
