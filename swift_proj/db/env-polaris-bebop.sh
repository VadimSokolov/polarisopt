module load readline/8.0-wejxbmz

export DB_HOST=${DB_HOST:-$(hostname --long)} # thetalogin4
# Use python to find a free port
PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export DB_PORT=${DB_PORT:-$PORT}
# The user name to use for the DB
export DB_USER=${DB_USER:-}
export DB_DATA=${DB_DATA:-/lcrc/project/EMEWS/bebop/db}
export DB_MODE=ON

export PATH=/lcrc/project/POLARIS/bebop/sfw/gcc-7.1.0/postgres-14.2/bin:$PATH
export LD_LIBRARY_PATH=/lcrc/project/POLARIS/bebop/sfw/gcc-7.1.0/postgres-14.2/lib:$LD_LIBRARY_PATH
