#/usr/bin/sh
# This is a script to copy files from one host to a group of hosts

# There are three variables accepted via commandline
# $1 = first parameter (/source_path/source_filename)
# $2 = second parameter (/target_directory/)
# $3 = third paramter (file that contains list of hosts)

SOURCEFILE=$1
#TARGETDIR=$2
#HOSTFILE=$3

if [ -f $SOURCEFILE ]
then
		while read node
		do
			scp -p $SOURCEFILE ${node}:$(pwd)
		done < "./hosts"
else
	printf "File \"$SOURCEFILE\" not found\n"
	exit 0
fi
exit 0
