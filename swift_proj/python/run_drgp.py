import sys

from PolarisOpt.setup_manager import SetupManager
from PolarisOpt.utils import util


def run(settings_fname, config_fname):
    manager = SetupManager(settings_fname, config_fname)
    util.build_sampleset(manager, '/home/nick/Documents/repos/polaris-hpc/swift_proj/data/samples.dat', num_samples=6)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
