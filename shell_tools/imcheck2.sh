#!/bin/bash
echo "Updating liveview image softlink every 2s"
while true
do 
	rm preview.bmp; 
	ls -d /mnt/ramdisk/*.bmp | tail -n 1 | xargs -I{} ln -s {} preview.bmp; 
	sleep 2; 
done

