#!/usr/bin/env bash
TITLE="`hostname`: all jobs are done."

calc () {
    echo - | awk "{print $1}"
}

while true; do
    count=`qstat | wc -l`
    if [[ $count -le 2 ]] ; then
        chmod 700 sendmail.py
        ./sendmail.py -u $EMAILUSER -p $EMAILPASS -r $EMAILUSER -s "$TITLE" -b "`date`"
        break
    fi
    echo `date`: remaining jobs: `calc ${count}-2`
    sleep 30
done