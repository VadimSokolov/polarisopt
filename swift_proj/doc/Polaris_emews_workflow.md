POLARIS EMEWS Workflow
----------------------

Argonne Bebop (linux)
---------------------

-   Setup and assumptions

    -   Bebop uses the module system to load all required software
        packages. EMEWS and POLARIS run/compilation have a specific set
        of dependencies.

        -   Source the bebop\_module.load.sh to load the required
            modules

    -   The \$emews\_root dir refers to the /polaris-linux/swift\_proj
        folder

    -   Use symbolic link \$emews\_root/data/polaris-data -\>
        polaris/data/

    -   Use symbolic link \$emews\_root/model/polaris -\>
        polaris/build/release/bin/Integrated\_model

    -   The file scenario\_init.json exists in the scenario folder

    -   POLARIS executable must be started in the scenario data folder

-   Unrolled Parameter File (UPF)

    -   Each line is a properly formatted json parameter set that will
        override the base scenario scenario\_init.json values, for
        example:

        -   { \"ending\_time\_hh\_mm\" : \"2:00\", \"seed\" : 1}

    -   Each line represents an individual model run

    -   We will generate the UPF via python or R to create the parameter
        space sweep scenario

-   bebop\_run\_polaris\_sweep.sh

    -   Main script that sets job params (PROCS, PPN, WALLTIME, etc)

    -   Requires arguments:

        -   experiment directory argument to write instance dirs.

        -   Scenario folder located in /polaris/data

    -   Calls polaris\_sweep.swift with the arguments:

        -   Unrolled parameter file (UPF) path

        -   Scenario folder name (not full path)

-   polaris\_sweep.swift

    -   Iterates over each UPF line and creates an instance folder for
        each

        -   Calls python script to read the POLARIS scenario\_init.json,
            modify parameters as defined in the UPF line, and then save
            the instance-specific scenario\_init.json in the instance
            folder.

        -   TODO read num threads from ~~UPF line~~ possible command
            line arg

        -   Executes /scripts/polaris.sh

-   polaris.sh

    -   script that runs the POLARIS executable

    -   Requires arguments:

        -   param line

        -   emews root

        -   instance directory

        -   scenario directory

        -   num\_threads

    -   CD into directory
        \$emews\_root/data/polaris-data/\$scenario\_dir

    -   Execute POLARIS:

        -   \"\$emews\_root/model/polaris
            \$instance\_directory/scenario\_init.json \$num\_threads\"

Windows 10 Subsystem for Linux (WSL)
------------------------------------

The process for running parameter sweeps with EMEWS with the Windows 10
linux subsystem is nearly identical to the Bebop workflow. The shell and
swift scripts require only minor modifications to specify the Windows
Polaris executable and addressing the volume label differences between
Windows and linux.

-   Setup and assumptions

    -   Using Windows 10 WSL and Ubuntu installed from the Microsoft
        Store.

    -   Using Polaris 0.8.1 Windows binary

    -   Use Spack in WSL to install swift-t and all depdencies

    -   EMEWS/Swift-t and shell scripts will run under WSL, but will
        call Windows binaries

    -   The \$emews\_root dir refers to the /polaris/swift\_proj folder

    -   Use symbolic link \$emews\_root/data/polaris-data -\>
        polaris/data/

    -   Use symbolic link \$emews\_root/model/polaris -\>
        polaris/bin/polaris.exe

    -   The file scenario\_init.json exists in the scenario folder

    -   POLARIS executable must be started in the scenario data folder

    -   WSL paths must follow the linux style volume labels, e.g.
        "/mnt/c"

    -   Windows file paths specified to the Polaris executable and
        scenario\_init.json must follow the Windows style volume labels,
        e.g. "C:"

### Installing EMEWS on Windows Linux Subsystem

-   Install GCC 7.x

    -   Add the ubuntu toolchain repo with newer GCC

        -   \$ sudo add-apt-repository ppa:ubuntu-toolchain-r/test

        -   \$ sudo apt-get update

    -   \$ sudo apt-get install gcc-7

    -   \$ sudo apt-get install g++-7

    -   Optionally remove gcc-x (x=4,5,6)

    -   \$ sudo apt-get remove gcc-x

-   Configure the gcc and g++ command to use ver.7 as the default

    -   \$ sudo update-alternatives \--install /usr/bin/gcc gcc
        /usr/bin/gcc-7 700 \--slave /usr/bin/g++ g++ /usr/bin/g++-7

-   Install FORTRAN w/ GCC 7 for compiling mpich. May not need fortran
    if other mpich install is used

    -   \$ sudo apt-get install gfortran-7

-   configure the f77 command to use ver 7 as default

    -   \$ sudo update-alternatives \--install /usr/bin/f77 f77
        /usr/bin/gfortran-7 700

-   Build mpich

    -   https://wiki.mpich.org/mpich/index.php/Getting\_And\_Building\_MPICH

    -   use only make - don\'t use 'make -j8', or use smaller thread num

    -   make sure to install in /usr/ not /usr/local

    -   \...otherwise cp all /usr/local/bin /lib /include mpi stuff to
        /usr

-   install a vanilla Python 2.7 for spack to use

    -   sudo apt-get install python

-   install curl for spack to use

    -   \$ sudo apt-get install curl

-   install spack - info
    <http://swift-lang.github.io/swift-t/guide.html#spack>

    -   \$ git clone <https://github.com/spack/spack.git>

    -   \$ . spack/share/spack/setup-env.sh

    -   \$ spack bootstrap

    -   Add the setup-env.sh to .profile or .bashrc

-   Configure spack to find all installed gcc

    -   \$ spack compiler find gcc

-   make sure gcc 7.4 is available

    -   \$ spack compiler info gcc

-   edit the ./spack/packages.yaml as needed

    -   specify compiler gcc 7.x so that all packages are built with the
        proper gcc

    -   let spack build tcl and python

-   Install swift-t with python

    -   \$ spack install stc \^turbine+python

    -   Optionally also specify +r to install R but make sure it's
        defined in the packages.yaml otherwise spack will try to build R
        which is a very length process.

-   load swift-t and verify configuration

    -   \$ spack load stc

    -   \$ swift-t -v

Example ./spack/packages.yaml
-----------------------------

Note that the line indentation is critical -- 2 spaces per level.

packages:

all:

compiler: \[gcc\@7.4.0\]

providers:

mpi: \[mpich\]

m4:

paths:

m4\@1.4.17 arch=linux-ubuntu18.04-x86\_64: /usr

buildable: False

mpich:

paths:

mpich\@3.3%gcc\@7.4.0 arch=linux-ubuntu18.04-x86\_64: /usr

buildable: False

jdk:

paths:

jdk\@1.8.0 arch=linux-ubuntu18.04-x86\_64: /usr

buildable: False

zsh:

paths:

zsh arch=linux-ubuntu18.04-x86\_64: /usr

buildable: False

swig:

paths:

swig\@3.0.8 arch=linux-ubuntu18.04-x86\_64: /usr

buildable: False

ant:

paths:

ant\@1.9.6 arch=linux-ubuntu18.04-x86\_64: /usr

buildable: False
