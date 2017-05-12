#!/bin/bash

while true; do
	ls -t /mnt/ramdisk/*.bmp | awk 'NR>10' | xargs -t -I '{}' mv {} /media/silcampc2/SlowData/raw/
	sleep 0.01
done
