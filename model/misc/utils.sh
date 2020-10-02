#!/usr/bin/env bash

calc () {
    echo - | awk "{print $1}"
}

elementIn () {
    local e match="$1"
    shift
    for e; do [[ "$e" == "$match" ]] && return 0; done
    return 1
}