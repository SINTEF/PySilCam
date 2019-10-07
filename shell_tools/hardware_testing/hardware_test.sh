#!/bin/bash
# runs test on hardware.
# Remember to change the paths in config file and the path to the data folder
# in the call to silcam realtime!

logfile=log-$(date +%Y-%m-%d--%H-%M-%S).txt
testtime=10 # number of seconds to run test
datafolder=/mnt/DATA/bjarnetest/DATA

echo "----TEST START: " $(date) "----" >> $logfile

echo "Clean DATA" >> $logfile
rm $datafolder/*.* >> $logfile
echo "  OK." >> $logfile

echo "START ACQUIRE: " $(date) >> $logfile
silcam realtime config_hardware_test.ini $datafolder --discwrite >> $logfile &

echo "waiting....." >> $logfile
sleep $testtime

echo "KILLING: " $(date) >> $logfile
killall silcam
echo "  kill sent: " $(date) >> $logfile

echo "ini files: " $(ls $datafolder/*.ini | wc -l) >> $logfile
echo "silc files: " $(ls $datafolder/*.silc | wc -l)>> $logfile
# @todo this doent work because of the time taken to initialise
#echo "  acquire freq.: "$(($(ls $datafolder/*.silc | wc -l)/$testtime)) >> $logfile
echo "bmp files: " $(ls $datafolder/*.bmp | wc -l)>> $logfile
# @todo this doent work because of the time taken to initialise
#echo "  acquire freq.: "$(($(ls $datafolder/*.bmp | wc -l)/$testtime)) >> $logfile


echo "STATS file: " $(ls $datafolder/proc/*STATS.csv) >> $logfile
echo "STATS length: " $(wc -l $datafolder/proc/*STATS.csv) >> $logfile

echo "exported files: " $(ls $datafolder/export/*.* | wc -l) >> $logfile
echo "----END----" >> $logfile

