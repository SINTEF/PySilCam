#!/bin/bash

while true; do
        cd /mnt/ramdisk/
        eog $(ls | tail -n 1)
        sleep 0.1
done
