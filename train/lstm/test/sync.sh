#!/usr/bin/env bash
#
# Run
#
RC=1
trap "exit" INT
while [[ RC -ne 0 ]]
do
  rsync -avzl -e "ssh -p8024" ananda@localhost:/Users/ananda/studio/dm/ ~/studio/dm/
	RC=$?
done
