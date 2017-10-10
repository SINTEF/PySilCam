import matplotlib, sys
matplotlib.use('TkAgg')
from numpy import arange, sin, pi
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from tkinter import *
import tkinter as tk
import skimage.io
import pickle
#from investigate_particles import *
import pysilcam.postprocess as scpp
from pysilcam.config import load_config, PySilcamSettings
import pandas as pd
import os
import numpy as np
from shutil import copyfile
import subprocess


DATADIR = '/mnt/ARRAY/ENTICE/Data/STN01'


class guiclass:
    def __init__(self, master):

        self.master = master
        master.title("Controller")
        
        self.lv = False

        self.toolbar = tk.Frame(self.master)
        self.quit = tk.Button(self.toolbar,
                text="Close",command=master.quit)
        self.quit.pack(side="left") # left side of parent, the toolbar frame

        self.recordbt = tk.Button(self.toolbar,
                text="RECORD",command=self.record)
        self.recordbt.pack(side="right") # left side of parent, the toolbar frame
        self.recordbt.configure(bg = "green")

        self.stopbt = tk.Button(self.toolbar,
                text="STOP RECORD",command=self.stop_record)
        self.stopbt.pack(side="right") # left side of parent, the toolbar frame
        self.stopbt.configure(bg = "blue")

        self.lvbt = tk.Button(self.toolbar,
                text="LV switch",command=self.lv_switch)
        self.lvbt.pack(side="left") # left side of parent, the toolbar frame
        self.lvbt.configure(bg = "blue")

        #self.build_buttons(classes)
#        self.other = tk.Button(self.toolbar,
#                text="Other",command=self.other)
#        self.other.pack(side="left") # left side of parent, the toolbar frame
#
#        self.copepod = tk.Button(self.toolbar,
#                text="Copepod",command=self.copepod)
#        self.copepod.pack(side="left") # left side of parent, the toolbar frame
#
#        self.diatom_chain = tk.Button(self.toolbar,
#                text="Diatom chain",command=self.diatom_chain)
#        self.diatom_chain.pack(side="left") # left side of parent, the toolbar frame

        f,self.a = plt.subplots(1,2,figsize=(5,5), dpi=100)
        self.dataPlot = FigureCanvasTkAgg(f, master=self.master)

        plt.sca(self.a[1])
        plt.axis('off')
        plt.sca(self.a[0])
        plt.axis('off')

        self.X_blank = np.zeros([1, 32, 32, 3],dtype='uint8')
        self.X = np.zeros([0, 32, 32, 3],dtype='uint8')
        self.Y = np.zeros((0,3),dtype='uint8')

        self.toolbar.pack(side=TOP, fill="x") # top of parent, the master window
        self.dataPlot.show()
        self.dataPlot.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        #self.canvas = Canvas(self, background="black")

        self.monitor()

    def monitor(self):

        #montxt = "ls /mnt/DATA/ | wc -l | awk '{{print $1}}'"
        #prc = subprocess.Popen([montxt], shell=True, stdout=subprocess.PIPE)
        #nimages = prc.stdout.read().decode('ascii').strip()
        files = [os.path.join(DATADIR, f) for f in sorted(os.listdir(DATADIR))]
        nimages = str(len(files))
        if len(files) > 3:

            name1 = os.path.split(files[-3])[1]
            ts1 = pd.to_datetime(name1[1:-4])
            name2 = os.path.split(files[-2])[1]
            ts2 = pd.to_datetime(name2[1:-4])
            td = ts2 - ts1
            td = td / np.timedelta64(1, 's')
            hz = 1 / td
            hz = str(np.around(hz, decimals=2))

            last_image = pd.to_datetime(pd.datetime.now()) - ts2
            last_image = last_image / np.timedelta64(1, 's')
            last_image = str(np.around(last_image, decimals=1))
            
        else:
            hz = 'waiting for data'
            last_image = ' waiting for data'
         

        montxt = "df -h | grep /mnt/DATA | awk '{{print $5}}'"
        prc = subprocess.Popen([montxt], shell=True, stdout=subprocess.PIPE)
        pcentfull = prc.stdout.read().decode('ascii').strip()


        ttlstr = (nimages + ' images\n\n' + 
            pcentfull + ' full\n\n' + hz + 'Hz\n\n' +
            last_image + ' sec. since prev.')

        plt.sca(self.a[1])
        plt.title(ttlstr, y=0.5, horizontalalignment='left')
        self.dataPlot.show()
        self.master.after(1*1000, self.monitor)



    def lv_switch(self):
        self.lv = np.invert(self.lv)
        self.update_image()

    def update_image(self):
        if not self.lv:
            self.lvbt.configure(bg = "blue")
            return
        self.lvbt.configure(bg = "green")
        try:
            files = [os.path.join(DATADIR, f) for f in sorted(os.listdir(DATADIR))]
            imfile = files[-4]

            name = os.path.split(imfile)[1]
            timestamp = pd.to_datetime(name[1:-4])
            plt.sca(self.a[0])

            im = skimage.io.imread(imfile)
            plt.cla()
            plt.imshow(im)
            plt.title(timestamp)
            plt.axis('off')
            plt.tight_layout()
            self.dataPlot.show()
        except ValueError:
            plt.title('error loading image')

        self.master.after(1000, self.update_image)

    def stop_record(self):
        self.recordbt.configure(bg = "blue", state=NORMAL)
        #subprocess.call('killall silcam-acquire', shell=True)

    def record(self):
        self.recordbt.configure(bg = "green", state=DISABLED)
        self.stopbt.configure(bg = "red")
        #self.process=subprocess.Popen(['pre xx'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True,preexec_fn=os.setsid)
        #self.process=subprocess.Popen(['source activate pysilcam; cd /mnt/DATA; silcam-acquire'], shell=True)
        #self.process=subprocess.Popen(['./logsilcam.sh'])
        

root = Tk()
my_gui = guiclass(root)
root.mainloop()

