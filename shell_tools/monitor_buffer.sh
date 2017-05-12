#!/bin/bash

while true; do
        date
        ls /mnt/ramdisk/ | wc -l | awk '{print $1 " images"}'
        df -h | grep ramdisk | awk '{print $5 " full"}'
	sleep 1
        clear
done
