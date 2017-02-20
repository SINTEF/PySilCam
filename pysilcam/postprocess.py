import pandas as pd
import numpy as np

PIX_SIZE = 35.2 / 2448 * 1000 # pixel size in microns (Med. mag)

def stats_from_csv(filename):
    stats = pd.read_csv(filename,index_col=0)
    return stats


def d50_from_stats(stats):
    ecd = stats['equivalent_diameter']
    ved = 4/3 * np.pi * (ecd/2)**3
    mved= np.median(ved)
    md = 2 * ((mved * 3) / (np.pi * 4))**(1/3)
    d50 = md * PIX_SIZE
    return d50


