#!/usr/bin/env bash

set -x

# doesn't actually cd to directory, but return absolute path
CUR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# cd to directory
cd $CUR_DIR

TIMEZONES_FILE=${TIMEZONES_FILE=$CUR_DIR/StorageSystemTimeZones.generated.cpp}

# if there are older files present, remove them so that fresh ones will be generated.
if [ ! -s $TIMEZONES_FILE.tmp || -s  $TIMEZONES_FILE ]; then
    echo "Removing existing file $TIMEZONES_FILE & $TIMEZONES_FILE.tmp"
    rm $TIMEZONES_FILE $TIMEZONES_FILE.tmp
fi

# Script will be run from $CUR_DIR i.e ClickHouse/src/Storages/System/StorageSystemTimeZones.sh
# Timezones are located under ClickHouse/contrib/cctz/testdata/zoneinfo so:
# * run find in target directory (Ignore *.tab files)
# * split by path '/' and extract timezone information
#   (i.e everything after ClickHouse/contrib/cctz/testdata/zoneinfo/testdata/ namely America/*, Africa/* etc...)
# * remove empty lines and sort
TIMEZONES=$(find  ../../../contrib/cctz/testdata/zoneinfo ! -name '*.tab' | cut -d'/' -f8- | sed '/^$/d' | sort)

# List of timezones are obtained from parsing the directory tree under ClickHouse/contrib/cctz/testdata/zoneinfo
# This is to remove top level dir names which are not valid timezones - ie. Pacific/Truk is a valid timezone but not Pacific.
for TZ in $TIMEZONES ;
do
  if [[ $TZ == */* ]] ;
  then
    echo '    "'$TZ'",' >> $TIMEZONES_FILE.tmp
  fi
done;

# If timezones are not available, don't generate the target file
if [ ! -s $TIMEZONES_FILE.tmp ]; then
    echo List of timezones are empty
    rm $TIMEZONES_FILE.tmp
    exit
fi

echo "// autogenerated by $0"             >  $TIMEZONES_FILE
echo "const char * auto_time_zones[] {"   >> $TIMEZONES_FILE
cat  $TIMEZONES_FILE.tmp                  >> $TIMEZONES_FILE
echo -e "    nullptr};"                   >> $TIMEZONES_FILE

echo "Collected `cat TIMEZONES_FILE.tmp | wc -l` timezones."
rm $TIMEZONES_FILE.tmp

