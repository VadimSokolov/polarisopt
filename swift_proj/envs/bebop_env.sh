# module load gcc/7.1.0-4bgguyp
# module load mvapich2/2.3a-avvw4kp 
# module load jdk
# module load anaconda3/2020.07

# export PATH=/lcrc/project/EMEWS/bebop/sfw/swift-t-083c396/stc/bin:$PATH

module load gcc/7.1.0-4bgguyp
module load mvapich2/2.3a-avvw4kp 
module load jdk
module load anaconda3/2020.07

# R_LIBS is set to some wonky path under spack's r-inside and rcpp for some reason. This should reset it.
export R_LIBS=/lcrc/project/EMEWS/bebop/repos/spack/opt/spack/linux-centos7-broadwell/gcc-7.1.0/r-4.0.0-plchfp7jukuhu5oity7ofscseg73tofx/rlib/R/library/
export PATH=/lcrc/project/EMEWS/bebop/sfw/swift-t-7771807/stc/bin:$PATH

