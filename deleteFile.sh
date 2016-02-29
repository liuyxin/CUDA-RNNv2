#/usr/bin/sh
# This is a script to copy files from one host to a group of hosts

# There are three variables accepted via commandline

SOURCEFILE=$1
#TARGETDIR=$2
#HOSTFILE=$3

		while read node
		do
			ssh ${node} "rm $(pwd)/$SOURCEFILE" < /dev/null
		done < "./hosts"
exit 0
