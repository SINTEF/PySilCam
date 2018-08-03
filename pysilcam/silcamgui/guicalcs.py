import pysilcam.__main__ as psc
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QMainWindow, QApplication, QPushButton, QWidget,
QAction, QTabWidget,QVBoxLayout, QFileDialog)
import os
from pysilcam.config import PySilcamSettings
import pysilcam.oilgas as scog
import numpy as np
import pysilcam.postprocess as sc_pp
import pandas as pd
from enum import Enum
import pygame
import time
import psutil
from tqdm import tqdm


def get_data(self):
    try:
        rts = self.q.get(timeout=0.1)
    except:
        rts = None
    return rts

def count_data(datadir):
    silcfiles = [os.path.join(datadir, f) for f in
            sorted(os.listdir(datadir))
            if f.endswith('.silc')]
    bmpfiles = [os.path.join(datadir, f) for f in
            sorted(os.listdir(datadir))
            if f.endswith('.bmp')]
    silc = len(silcfiles)
    bmp = len(bmpfiles)
    return silc, bmp

def extract_stats_im(guidata):
    imc = guidata['imc']
    del guidata['imc']
    stats = pd.DataFrame.from_dict(guidata)
    return stats, imc


def export_timeseries(configfile, statsfile):

    settings = PySilcamSettings(configfile)

    print('Loading STATS data: ', statsfile)
    stats = pd.read_csv(statsfile)

    stats['timestamp'] = pd.to_datetime(stats['timestamp'])

    stats.sort_values(by='timestamp', inplace=True)

    print('Extracting oil and gas')
    stats_oil = scog.extract_oil(stats)
    stats_gas = scog.extract_gas(stats)

    print('Calculating timeseries')
    u = pd.to_datetime(stats['timestamp']).unique()

    sample_volume = sc_pp.get_sample_volume(settings.PostProcess.pix_size, path_length=settings.PostProcess.path_length)

    td = pd.to_timedelta('00:00:' + str(settings.PostProcess.window_size / 2.))

    vdts_all = []
    vdts_oil = []
    vdts_gas = []
    d50_all = []
    d50_oil = []
    d50_gas = []
    timestamp = []
    d50_av_all = []
    d50_av_oil = []
    d50_av_gas = []
    gor = []
    for s in tqdm(u):
        timestamp.append(pd.to_datetime(s))
        dt = pd.to_datetime(s)

        dias, vd_all = sc_pp.vd_from_stats(stats[stats['timestamp'] == s],
                                 settings.PostProcess)
        dias, vd_oil = sc_pp.vd_from_stats(stats_oil[stats_oil['timestamp'] == s],
                                 settings.PostProcess)
        dias, vd_gas = sc_pp.vd_from_stats(stats_gas[stats_gas['timestamp'] == s],
                                 settings.PostProcess)

        nims = sc_pp.count_images_in_stats(stats[stats['timestamp'] == s])
        sv = sample_volume * nims
        vd_all /= sv
        vd_oil /= sv
        vd_gas /= sv
        d50_all.append(sc_pp.d50_from_vd(vd_all, dias))
        d50_oil.append(sc_pp.d50_from_vd(vd_oil, dias))
        d50_gas.append(sc_pp.d50_from_vd(vd_gas, dias))

        vdts_all.append(vd_all)
        vdts_oil.append(vd_oil)
        vdts_gas.append(vd_gas)

        stats_av = stats[(stats['timestamp']<(dt+td)) & (stats['timestamp']>(dt-td))]
        stats_av_oil = scog.extract_oil(stats_av)
        stats_av_gas = scog.extract_gas(stats_av)
        d50_av_all.append(sc_pp.d50_from_stats(stats_av, settings.PostProcess))
        d50_av_oil.append(sc_pp.d50_from_stats(stats_av_oil, settings.PostProcess))
        d50_av_gas.append(sc_pp.d50_from_stats(stats_av_gas, settings.PostProcess))

        dias, vdts_av = sc_pp.vd_from_stats(stats_av, settings.PostProcess)
        dias, vdts_av_oil = sc_pp.vd_from_stats(stats_av_oil, settings.PostProcess)
        dias, vdts_av_gas = sc_pp.vd_from_stats(stats_av_gas, settings.PostProcess)
        nims = sc_pp.count_images_in_stats(stats_av)
        sv = sample_volume * nims
        vdts_av /= sv
        vdts_av_oil /= sv
        vdts_av_gas /= sv

        gor.append(np.sum(vdts_av_gas)/np.sum(vdts_av_oil))

    outpath, outfile = os.path.split(statsfile)
    outfile = outfile.replace('-STATS.csv','')
    outfile = os.path.join(outpath, outfile)

    time_series = pd.DataFrame(data=np.squeeze(vdts_all), columns=dias)
    time_series['D50'] = d50_all
    time_series['Time'] = timestamp
    time_series.to_excel(outfile +
            '-TIMESERIES' + '' + '.xlsx')

    time_series = pd.DataFrame(data=np.squeeze(vdts_oil), columns=dias)
    time_series['D50'] = d50_all
    time_series['Time'] = timestamp
    time_series.to_excel(outfile +
            '-TIMESERIES' + 'oil' + '.xlsx')

    time_series = pd.DataFrame(data=np.squeeze(vdts_gas), columns=dias)
    time_series['D50'] = d50_all
    time_series['Time'] = timestamp
    time_series.to_excel(outfile +
            '-TIMESERIES' + 'gas' + '.xlsx')

    plt.figure(figsize=(20, 10))

    if not np.min(np.isnan(d50_oil)):
        plt.plot(timestamp, d50_oil, 'ro')
    if not np.min(np.isnan(d50_av_oil)):
        plt.plot(timestamp, d50_av_oil, 'r-')
    lns1 = plt.plot(np.nan, np.nan, 'r-', label='OIL')

    if not np.min(np.isnan(d50_gas)):
        plt.plot(timestamp, d50_gas, 'bo')
    if not np.min(np.isnan(d50_av_gas)):
        plt.plot(timestamp, d50_av_gas, 'b-')
    lns2 = plt.plot(np.nan, np.nan, 'b-', label='GAS')

    plt.ylabel('d50 [um]')
    plt.ylim(0, max(plt.gca().get_ylim()))

    ax = plt.gca().twinx()
    plt.sca(ax)
    plt.ylabel('GOR')
    if not np.min(np.isnan(gor)):
        plt.plot(timestamp, gor, 'k')
    lns3 = plt.plot(np.nan, np.nan, 'k', label='GOR')
    plt.ylim(0, max(plt.gca().get_ylim()))

    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs)

    plt.savefig(outfile +
                '-d50_TimeSeries.png', dpi=600, bbox_inches='tight')

    plt.close()
    print('Export figure made. ')
    print('Exporting averages... ')

    # average all
    dias, vd = sc_pp.vd_from_stats(stats,
                             settings.PostProcess)
    nims = sc_pp.count_images_in_stats(stats)
    sv = sample_volume * nims
    vd /= sv
    d50 = sc_pp.d50_from_vd(vd, dias)
    dfa = pd.DataFrame(data=[vd], columns=dias)
    dfa['d50'] = d50
    timestamp = np.min(pd.to_datetime(stats['timestamp']))
    dfa['Time'] = timestamp
    dfa.to_excel(statsfile.replace('-STATS.csv', '') +
                 '-AVERAGE' + '' + '.xlsx')

    #average oil
    dias, vd = sc_pp.vd_from_stats(stats_oil,
                             settings.PostProcess)
    vd /= sv # sample volume remains the same as 'all'
    d50 = sc_pp.d50_from_vd(vd, dias)
    dfa = pd.DataFrame(data=[vd], columns=dias)
    dfa['d50'] = d50
    timestamp = np.min(pd.to_datetime(stats['timestamp'])) # still use total stats for this time
    dfa['Time'] = timestamp
    dfa.to_excel(statsfile.replace('-STATS.csv', '') +
                 '-AVERAGE' + 'oil' + '.xlsx')

    #average gas
    dias, vd = sc_pp.vd_from_stats(stats_gas,
                             settings.PostProcess)
    vd /= sv # sample volume remains the same as 'all'
    d50 = sc_pp.d50_from_vd(vd, dias)
    dfa = pd.DataFrame(data=[vd], columns=dias)
    dfa['d50'] = d50
    timestamp = np.min(pd.to_datetime(stats['timestamp'])) # still use total stats for this time
    dfa['Time'] = timestamp
    dfa.to_excel(statsfile.replace('-STATS.csv', '') +
                 '-AVERAGE' + 'gas' + '.xlsx')

    print('Export done: ', outfile)


def load_image(filename, size):
    if filename.endswith('.silc'):
        with open(filename, 'rb') as fh:
            im = np.load(fh, allow_pickle=False)
        im = pygame.surfarray.make_surface(np.uint8(im))
        im = pygame.transform.flip(im, False, True)
        im = pygame.transform.rotate(im, -90)
        im = pygame.transform.scale(im, size)
    else:
        im = pygame.image.load(filename).convert()

    return im


def silcview(datadir):
    files = [os.path.join(datadir, f) for f in
            sorted(os.listdir(datadir))
            if f.endswith('.silc') or f.endswith('.bmp')]
    if len(files)==0:
        return
    pygame.init()
    info = pygame.display.Info()
    size = (int(info.current_h / (2048/2448))-100, info.current_h-100)
    screen = pygame.display.set_mode(size)
    font = pygame.font.SysFont("monospace", 20)
    c = pygame.time.Clock()
    zoom = False
    counter = -1
    direction = 1 # 1=forward 2=backward
    pause = False
    pygame.event.set_blocked(pygame.MOUSEMOTION)
    while True:
        if pause:
            event = pygame.event.wait()
            if event.type == 12:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    zoom = np.invert(zoom)
                if event.key == pygame.K_LEFT:
                    direction = -1
                if event.key == pygame.K_RIGHT:
                    direction = 1
                if event.key == pygame.K_p:
                    pause = np.invert(pause)
                else:
                    continue
                pygame.time.wait(100)

        counter += direction
        counter = np.max([counter, 0])
        counter = np.min([len(files)-1, counter])
        c.tick(15) # restrict to 15Hz
        f = files[counter]

        if not (counter == 0 | counter==len(files)-1):
            filename = os.path.join(datadir, f)
            im = load_image(filename, size)

        if zoom:
            label = font.render('ZOOM [F]: ON', 1, (255, 255, 0))
            im = pygame.transform.scale2x(im)
            screen.blit(im,(-size[0]/2,-size[1]/2))
        else:
           im = pygame.transform.scale(im, size)
           screen.blit(im,(0,0))
           label = font.render('ZOOM [F]: OFF', 1, (255, 255, 0))
        screen.blit(label,(0, size[1]-20))

        if direction==1:
            dirtxt = '>>'
        elif direction==-1:
            dirtxt = '<<'
        if pause:
            dirtxt = 'PAUSED ' + dirtxt
        label = font.render('DIRECTION [<-|->|p]: ' + dirtxt, 1, (255,255,0))
        screen.blit(label, (0, size[1]-40))

        if counter == 0:
            label = font.render('FIRST IMAGE', 1, (255,255,0))
            screen.blit(label, (0, size[1]-60))
        elif counter == len(files)-1:
            label = font.render('LAST IMAGE', 1, (255,255,0))
            screen.blit(label, (0, size[1]-60))

        timestamp = pd.to_datetime(
                os.path.splitext(os.path.split(f)[-1])[0][1:])


        pygame.display.set_caption('raw image replay:' + os.path.split(f)[0])#, icontitle=None)
        label = font.render(str(timestamp), 20, (255, 255, 0))
        screen.blit(label,(0,0))
        label = font.render('Display FPS:' + str(c.get_fps()),
                1, (255, 255, 0))
        screen.blit(label,(0,20))

        for event in pygame.event.get():
            if event.type == 12:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    zoom = np.invert(zoom)
                if event.key == pygame.K_LEFT:
                    direction = -1
                if event.key == pygame.K_RIGHT:
                    direction = 1
                if event.key == pygame.K_p:
                    pause = np.invert(pause)
                    direction = 0

        pygame.display.flip()

    pygame.quit()


class process_mode(Enum):
    process = 1
    aquire = 2
    real_time = 3


class ProcThread(Process):
    run_type = process_mode.process

    def __init__(self, datadir, configfile, disc_write, run_type, overwriteSTATS, fighandle):
        super(ProcThread, self).__init__()
        self.q = Queue(1)
        self.info = 'ini done'
        self.datadir = datadir
        self.configfile = configfile
        self.settings = ''
        self.rts = ''
        self.disc_write = disc_write
        self.run_type = run_type
        self.overwriteSTATS = overwriteSTATS
        self.fighandle = fighandle


    def run(self):
        if(self.run_type == process_mode.process):
            psc.silcam_process(self.configfile, self.datadir, multiProcess=True, realtime=False,
            gui=self.q, overwriteSTATS=self.overwriteSTATS)
        elif(self.run_type == process_mode.aquire):
            psc.silcam_acquire(self.datadir, config_filename=self.configfile, writeToDisk=self.disc_write, gui=self.q)
        elif(self.run_type == process_mode.real_time):
            psc.silcam_process(self.configfile, self.datadir, multiProcess=True, realtime=True,
                               discWrite=self.disc_write, gui=self.q, overwriteSTATS=self.overwriteSTATS)

        #psc.silcam_sim(self.datadir, self.q)


    def go(self):
        self.start()


    def stop_silcam(self):

        if self.is_alive():
            self.terminate()
            # terminate all children processes
            list = psutil.Process(self.pid).children()
            for p in list:
                p.terminate()
            self.join()
            self.info = 'termination sent'
        else:
            self.info = 'nothing to terminate'


    def plot(self):
        infostr = 'waiting to plot'
        if self.rts == '':
            self.rts = scog.rt_stats(self.settings)


        if self.is_alive():
            guidata = get_data(self)
            if not guidata == None:
                #stats, imc = extract_stats_im(guidata)
                timestamp = guidata[0]
                imc = guidata[1]
                imraw = guidata[2]
                dias = guidata[3]['dias']
                vd_oil = guidata[3]['vd_oil']
                vd_gas = guidata[3]['vd_gas']
                oil_d50 = guidata[3]['oil_d50']
                gas_d50 = guidata[3]['gas_d50']
                saturation = guidata[3]['saturation']
                gor = np.float64(np.sum(vd_gas)/np.sum(vd_oil))

                #infostr = data['infostr']
                infostr = 'got data'

                plt.figure(self.fighandle)
                plt.clf()
                plt.cla()


                plt.subplot(2,2,1)
                plt.cla()
                plt.plot(dias, vd_oil ,'r')
                plt.plot(dias, vd_gas ,'b')
                plt.xscale('log')
                plt.xlim((50, 12000))
                plt.ylabel('Volume Concentration [uL/L]')
                plt.xlabel('Diameter [um]')

                plt.subplot(2,2,3)
                plt.cla()
                ttlstr = (
                        'Oil d50: {:0.0f} [um]'.format(oil_d50) + '\n' +
                        'Gas d50: {:0.0f} [um]'.format(gas_d50) + '\n' +
                        'GOR: {:0.2f}'.format(gor) + ' ' + ' Saturation: {:0.0f} [%]'.format(saturation)
                        )
                plt.title(ttlstr)
                plt.imshow(imraw)
                plt.axis('off')

                plt.subplot(1,2,2)
                ttlstr = ('Image time: ' +
                    str(timestamp))
                plt.cla()
                plt.imshow(imc)
                plt.axis('off')
                plt.title(ttlstr)

                plt.tight_layout()

            self.info = infostr


    def load_settings(self, configfile):
        self.settings = PySilcamSettings(configfile)

