
module load gcc/7.1.0-4bgguyp 
module load mvapich2/2.3a-avvw4kp

module unload intel-mkl/2018.1.163-4okndez
. /lcrc/project/EMEWS/bebop/repos/spack/share/spack/setup-env.sh
# r 4.0.0
spack load /plchfp7
spack load intel-mkl@2020.1.217

# export R_LIBS=/lcrc/project/EMEWS/bebop/repos/spack/opt/spack/linux-centos7-broadwell/gcc-7.1.0/r-4.0.0-plchfp7jukuhu5oity7ofscseg73tofx/rlib/R/library/
export PYTHONHOME=/lcrc/project/EMEWS/bebop/sfw/anaconda3/2020.11
export PATH=$PYTHONHOME/bin:/lcrc/project/POLARIS/ncollier/sfw/gcc-7.1.0/mvapich2.3/swift-t-dfb7c62/stc/bin:$PATH
