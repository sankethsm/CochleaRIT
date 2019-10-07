"""
Argument parser for model
"""

import argparse

#-------------------------------------------------------------------------------#

def argParser():
    """
    Custom argument parser for the model
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=1)

    parser.add_argument('--nEpochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--mm', type=float, default=0.9)

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--wd', default=1e-6, type=float,
                        help='weight decay (default: 1e-8)')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--save')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam'))

    return parser