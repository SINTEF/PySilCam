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


DATABASE_PATH = '/mnt/ARRAY/silcam_classification_database'
config_file = '/mnt/ARRAY/ENTICE/Data/configs/config.ini'
stats_file = '/mnt/ARRAY/ENTICE/Data/proc/STN10-STATS.h5'
filepath = '/mnt/ARRAY/ENTICE/Data/export/'

def find_classes(d=DATABASE_PATH):
    classes = [o for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
    print(classes)
    return classes

def particle_generator():
    conf = load_config(config_file)
    settings = PySilcamSettings(conf)

    stats = pd.read_hdf(stats_file, 'ParticleStats/stats')

    print('all stats:', len(stats))

    index = 0

    while True:

#        if np.random.choice([0,1]):
        stats_ = scpp.extract_nth_largest(stats, n=index)
#        else:
#           stats_ = scpp.extract_nth_longest(stats,settings,n=index)
        print(stats_)
        filename = os.path.join(filepath, stats_['export name'])

        im = skimage.io.imread(filename)

        im = scpp.explode_contrast(im)
        im = scpp.bright_norm(im)
        # scplt.show_imc(im)
        # plt.title(selected_stats['export name'] + ('\nDepth:
        # {0:.1f}'.format(selected_stats['depth'])) + 'm\n')

        index += 1

        filename = stats_['export name']

        yield im, filepath, filename


class guiclass:
    def __init__(self, master):
        classes = find_classes()

        self.master = master
        master.title("Class builder")
        

        self.toolbar = tk.Frame(self.master)
        self.quit = tk.Button(self.toolbar,
                text="Close",command=master.quit)
        self.quit.pack(side="left") # left side of parent, the toolbar frame

        self.next = tk.Button(self.toolbar,
                text="Next",command=self.update_image)
        self.next.pack(side="right") # left side of parent, the toolbar frame

        self.choice = StringVar(self.master)
        self.choice.set(classes[0])
        self.w = OptionMenu(self.toolbar, self.choice, *classes)
        self.w.pack(side='right')

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

        f,self.a = plt.subplots(1,1,figsize=(5,5), dpi=100)
        self.dataPlot = FigureCanvasTkAgg(f, master=self.master)

        self.pgen = particle_generator()
        self.im, self.imfilepath, self.imfilename = next(self.pgen)
        plt.sca(self.a)
        plt.imshow(self.im)
        plt.axis('off')

        self.X_blank = np.zeros([1, 32, 32, 3],dtype='uint8')
        self.X = np.zeros([0, 32, 32, 3],dtype='uint8')
        self.Y = np.zeros((0,3),dtype='uint8')

        self.toolbar.pack(side=TOP, fill="x") # top of parent, the master window
        self.dataPlot.show()
        self.dataPlot.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

    def update_image(self):
        self.dump_data()
        plt.sca(self.a)
        plt.cla()
        self.im, self.imfilepath, self.imfilename = next(self.pgen)
        plt.imshow(self.im)
        plt.axis('off')
        self.dataPlot.show()


    def dump_data(self):
        choice = self.choice.get()
        print('from:')
        print(os.path.join(self.imfilepath, self.imfilename))
        print('to:')
        print(os.path.join(DATABASE_PATH, choice, self.imfilename))
        copyfile(os.path.join(self.imfilepath, self.imfilename),
                os.path.join(DATABASE_PATH, choice, self.imfilename))

        return

root = Tk()
my_gui = guiclass(root)
root.mainloop()

