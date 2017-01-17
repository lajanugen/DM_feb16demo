#!/usr/bin/python
# -*- coding: utf-8 -*-

from optparse import OptionParser
import sys, pdb, traceback, os
from math import log10
import numpy as np, re
from multiprocessing import Pool, cpu_count

from src import utils, plot
timer = utils.runtimetracker()

def parse_args():
  parser = OptionParser(usage=" ", add_help_option=False)
  parser.add_option("-h", "--help", action="help",
      help="Show this help message and exit")

  (opts, args) = parser.parse_args()
  if len(args) != 0:
    parser.print_help()
    sys.exit(1)

  return (opts, args)

def find_best_inds(perfs):
  flatind = np.argmin(perfs)
  i, k = np.unravel_index(flatind, perfs.shape)
  return (i, k)

def main():
  fnames = ['final_dv2-512-lr0.02.txt', 'final_dv2-512-lr0.002.txt', 'final_dv2-128-lr0.02.txt', 'final_dv2-128-lr0.002.txt']
  tot_models = len(fnames)

  num_surr = 3
  avgT = np.zeros((tot_models,num_surr))
  ebarT = np.zeros((tot_models,num_surr))
  avgVal = np.zeros((tot_models,num_surr))
  ebarVal = np.zeros((tot_models,num_surr))

  for i,fname in enumerate(fnames):
    obj = utils.read_jsonfile('results/%s' % fname)

    for j in range(num_surr):
      avgVal[i,j] = obj['best_val_perf'][j]['avg']*100
      ebarVal[i,j] = obj['best_val_perf'][j]['conf_intr']*100
      avgT[i,j] = obj['best_test_perf'][j]['avg']*100
      ebarT[i,j] = obj['best_test_perf'][j]['conf_intr']*100

  best_ind = np.argmax(avgVal[:,0]) # refers to accuracy, the first surrogate measure
  print 'Best model: %s' % fnames[best_ind]

  avgF = np.zeros((1,num_surr))
  ebarF = np.zeros((1,num_surr))
  avgF[0,:] = avgT[best_ind,:]
  ebarF[0,:] = ebarT[best_ind,:]
  bsavg = None # avgVal[best_ind,0]
  bsebar = None # ebarVal[best_ind,0]

  # avgF = avgT
  # ebarF = ebarT
  # pdb.set_trace()
  itr = 1
  plot.plot_ebar(avgF, ebarF, xticklabels=[''], xlabel='Model-1', ylabel='(in %)',
    title='Surrogate performance measures on Test set\n(higher better)\n',
    fname='results/lstm1/%d.png' % itr, labels=['accuracy', 'precision', 'recall'], bsavg=bsavg, bsebar=bsebar, width=0.06, aspect=None)

  # pdb.set_trace()
  return

if  __name__ =='__main__':
  try:
    main()
  except:
    type, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)
