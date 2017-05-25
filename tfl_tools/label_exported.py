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
from investigate_particles import *



class guiclass:
    def __init__(self, master):
        self.master = master
        master.title("Title txt")

        self.toolbar = tk.Frame(self.master)
        self.quit = tk.Button(self.toolbar,
                text="Close",command=master.quit)
        self.quit.pack(side="left") # left side of parent, the toolbar frame

        self.other = tk.Button(self.toolbar,
                text="Other",command=self.other)
        self.other.pack(side="left") # left side of parent, the toolbar frame

        self.copepod = tk.Button(self.toolbar,
                text="Copepod",command=self.copepod)
        self.copepod.pack(side="left") # left side of parent, the toolbar frame

        self.diatom_chain = tk.Button(self.toolbar,
                text="Diatom chain",command=self.diatom_chain)
        self.diatom_chain.pack(side="left") # left side of parent, the toolbar frame
  

        f,self.a = plt.subplots(1,1,figsize=(5,5), dpi=100)
        self.dataPlot = FigureCanvasTkAgg(f, master=self.master)

        self.pgen = particle_generator()
        self.im = next(self.pgen)
        plt.sca(self.a)
        plt.imshow(self.im)
        plt.axis('off')

        self.X_blank = np.zeros([1, 32, 32, 3],dtype='uint8')
        self.X = np.zeros([0, 32, 32, 3],dtype='uint8')
        self.Y = np.zeros((0,3),dtype='uint8')

        self.toolbar.pack(side=TOP, fill="x") # top of parent, the master window
        self.dataPlot.show()
        self.dataPlot.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)


    def add_im_to_stack(self):
        imrs = skimage.transform.resize(self.im, (32, 32, 3), mode='reflect',
                preserve_range=True)
        imrs = np.uint8(imrs)
            
        self.X = np.vstack((self.X, self.X_blank))
        self.X[-1,:] = imrs


    def other(self):
        y = [1, 0, 0]
        self.Y = np.vstack((self.Y, y))
        self.add_im_to_stack()
        self.update_image()


    def copepod(self):
        y = [0, 1, 0]
        self.Y = np.vstack((self.Y, y))
        self.add_im_to_stack()
        self.update_image()
        

    def diatom_chain(self):
        y = [0, 0, 1]
        self.Y = np.vstack((self.Y, y))
        self.add_im_to_stack()
        self.update_image()


    def update_image(self):
        self.dump_data()
        plt.sca(self.a)
        plt.cla()
        self.im = next(self.pgen)
        plt.imshow(self.im)
        plt.axis('off')
        self.dataPlot.show()


    def dump_data(self):
        fid = open('DATA_12.pkl','wb')
        pickle.dump((self.X,self.Y),fid)

root = Tk()
my_gui = guiclass(root)
root.mainloop()
