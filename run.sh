#/usr/bin/sh
mpiexec -f ./machinefile -n $1 $(pwd)/$2
exit 0
