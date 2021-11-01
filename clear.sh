#!/bin/bash

pids=$(pgrep -u ${USER} -a | grep "from multiprocessing.spawn import spawn_main" | cut -d' ' -f1)
for pid in ${pids}; do
    kill -9 ${pid} > /dev/null 2>&1
done