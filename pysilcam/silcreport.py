import matplotlib
matplotlib.use('Agg')
import pysilcam.plotting as scplt
import pysilcam.postprocess as scpp
from docopt import docopt
import matplotlib.pyplot as plt

def silcreport():
    '''Generate a report figure for a processed dataset from the SilCam

    Usage:
        silcam-report <configfile> <statsfile> [--type=<particle_type>]
                      [--dpi=<dpi>] [--monitor]

    Arguments:
        configfile  The config filename associated with the data
        statsfile   The -STATS.csv filename associated with the data

    Options:
        --type=<particle_type>  The particle type to summarise. Can be: 'all',
                                'oil', or 'gas'
        --dpi=<dpi>             DPI resolution of figure (default is 600)
        --monitor               Enables continuous monitoring (requires display)
        -h --help               Show this screen.

    '''

    args = docopt(silcreport.__doc__)
    silcam_report(args, monitor=False, dpi=600)

def silcam_report(args, monitor=False, dpi=600):
    '''does reporting'''
    particle_type = scpp.outputPartType.all
    particle_type_str = 'all'
    if args['--type']=='oil':
        particle_type = scpp.outputPartType.oil
        particle_type_str = args['--type']
    elif args['--type']=='gas':
        particle_type = scpp.outputPartType.gas
        particle_type_str = args['--type']

    if args['--dpi']:
        dpi=int(args['--dpi'])

    if args['--monitor']:
        print('  Monitoring enabled:')
        print('    press ctrl+c to stop.')
        monitor=True

    plt.figure(figsize=(20,12))

    scplt.summarise_fancy_stats(args['<statsfile>'], args['<configfile>'],
            monitor=monitor, oilgas=particle_type)

    print('  Saving to disc....')
    plt.savefig(args['<statsfile>'].strip('-STATS.csv') + '-Summary_' +
            particle_type_str + '.png',
            dpi=dpi, bbox_inches='tight')
    print('Done.')




