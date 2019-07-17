Linux installation on Windows 10
================================

1.  Install WSL Ubuntu 18.04 LTS on Windows 10 machine

2.  Make sure the to install the following in Ubuntu:

-   Update the installation repositories first:

    -   sudo apt-get update 

-   Install the build tools:

    -   sudo apt install build-essential

-   Git (should be installed on Ubuntu 18.04)

-   Python3, including PIP for version 3. You need to use the
    environment structure mentioned here:

    -   <https://stackoverflow.com/questions/10763440/how-to-install-python3-version-of-package-via-pip-on-ubuntu>

	    -   You may need to install virtualenv first:
	    	-   sudo install virtualenv
	    -   virtualenv -p /usr/bin/python3 py3env
	    -   source py3env/bin/activate

    
-   Install the parsl package in python:

    -   Make sure you are using the correct version of pip: 
	- pip --version

    -   Install the python development headers:
	- sudo apt-get install python3.6-dev

    -   Install the parsl package
	- pip install parsl

3.  Mapping to the network drive makes things much easier:

    a.  <https://superuser.com/questions/1128634/how-to-access-network-mounted-drive-on-windows-linux-subsystem/1261563>

    b.  Use:

        i.  sudo mkdir /mnt/p

        ii. sudo mount -t drvfs '\\vms-fs\VMS' /mnt/p

        iii. sudo mkdir /mnt/h

        iv. sudo mount -t drvfs '\\ascend.egs.anl.gov\groups\AMD-ES-Shared' /mnt/h

LCRC Resource Settings
======================

1.  Create LCRC account

    <http://www.lcrc.anl.gov/for-users/getting-started/getting-an-account/>

2.  Set up SSH keys and register

    <http://www.lcrc.anl.gov/for-users/getting-started/ssh/openssh/>

> NOTE:
>
> Do NOT set up you SSH public key using a pass phrase!!!!!
>
> You will not be able to input the pass phrase from the script and you
> will not be able to submit jobs. You need to send an email to support
> to get the ssh keys registered, so allow time for this step.

3.  Set default project for your LCRC user name (from Bebop)

    <http://www.lcrc.anl.gov/for-users/using-lcrc/running-jobs/running-jobs-on-bebop/>

    lcrc-sbank -s default POLARIS

Clone Repository and Execute Scripts
====================================

1.  Clone the HPC project from git. You need the client side scripts
    from the ZOV/Cplex project:

    git clone <https://github.com/anl-polaris/polaris-hpc.git>

2.  Navigate to the client side scripts folder and execute the script to
    run the cplex process:

    cd Polaris-hpc/zov\_cplex\_proj/client\_side\_scripts

    ./run\_cplex.sh \<path to base directory\>

    Example:

    ./run\_cplex.sh /mnt/h/POLARIS/Scenario\_C\_EVIPRO/chicago\_C\_highT

    This will manipulate the files in the ZOV folder under the base
    directory and create the following directory structure:

    /mnt/h/POLARIS/Scenario\_C\_EVIPRO/chicago\_C\_highT

    -   run\_\<timestamp\>

        inputs

        results

        When the script finishes you will have the results you need in
        the results folder
