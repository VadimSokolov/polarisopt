import proxies
import json
import sys
from PolarisOpt import eval_sim


def eval(f_str, proxy_str, params_str):
    f = proxies.load_proxies({'f': json.loads(f_str)})['f']
    return f(proxy_str, params_str)


@proxies.app
def eval_sample_task(manager, output_fp, pend_samples, row: int):
    pend_sample = pend_samples[row]
    eval_sim.eval_sample_task(manager, output_fp, pend_sample, row)


@proxies.app
def eval_dr_task(manager, dr_model, pend_samples, row: int):
    pend_sample = pend_samples[row]
    eval_sim.eval_DR_task(manager, dr_model, pend_sample, row)


if __name__ == '__main__':
    f_str = sys.argv[1]
    proxy_str = sys.argv[2]
    params_str = sys.argv[3]
    eval(f_str, proxy_str, params_str)
