#!/usr/bin/python
# -*- coding: utf-8 -*-

from math import isnan
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpltools import style, layout
style.use('ggplot')
from astroML.plotting import scatter_contour

import pandas, pdb, numpy as np
import scipy, json, sys, traceback
import scipy.stats
from collections import Counter
# import pylab
# from src import data

MAXUNIQS_FULLPIE = 12
# MAXPERCT_PARTPIE = 0.
MAXUNIQS_FULLBAR = 40
# MAXPERCT_PARTBAR = 0.
MAXPERCT_ENTROPY = 100.
MAXPERCT_TOPKENTROPY = 90.

MAXPERCTENT_SCATTER = 50.


# TODO: For every Pie chart, we could instead plot a bar graph, and vice versa

def heatmap(np_array, xticks, yticks, fname='/tmp/tmp.png', xlabel='', ylabel='', title=''):
  plt.imshow(np_array, interpolation='none', aspect='equal')
  plt.xticks(range(len(xticks)), xticks)
  plt.yticks(range(len(yticks)), yticks)
  plt.jet()
  plt.colorbar()
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  print_(fname)
  # plt.show()

def plot_heatmap_validation(pars_1, pars_2, perfs, fname):
  # pars_1 and pars_2 are list of two changing parameters
  # perfs is a dictionary of key as (par_1, par_2) tuple, and value being the score/performance

  perfs_np = np.zeros((len(pars_1),len(pars_2)))
  for i,v1 in enumerate(pars_1):
    for j,v2 in enumerate(pars_2):
      perfs_np[i,j] = perfs[(v1,v2)]

  x_lab = np.around(pars_2, decimals=1)
  y_lab = np.around(pars_1, decimals=1)
  heatmap(perfs_np, x_lab, y_lab, fname)


def print_(fname):
  plt.savefig(fname, dpi=300, bbox_inches='tight') #
  plt.clf()


def plot_ebar(avgs, ebars, xticklabels, xlabel, ylabel, title, fname, labels=None, bsavg=None, bsebar=None, width=0.35, aspect=None, fontsize=12):
  # avgs and ebars are both numpy arrays of size n by k, where we're comparing k different models.
  # for eg., if they're size 6 x 2, we're plotting 6 bars, but for each bar, we're comparing 2 models
  # in this eg, xticklabels will be a list of size 6, and labels will be a list of size 2

  # aspect can be 'auto' or 'equal'; use 'equal' for lengthy plots
  # width is the width of the bars

  # if its a squeezed np matrix, reshape
  if len(avgs.shape)==1:
    avgs = np.reshape(avgs,(avgs.shape[0],1))
  if len(ebars.shape)==1:
    ebars = np.reshape(ebars,(ebars.shape[0],1))

  ind = width+np.arange(avgs.shape[0])  # the x locations for the groups

  all_colors = plt.rcParams['axes.color_cycle']
  fig, ax = plt.subplots()
  rects_all = []
  for i in range(avgs.shape[1]):
    color = all_colors[i%len(all_colors)]
    if i<len(all_colors):
      hatch=None
    else:
      hatch='/'
    rects_all.append(ax.bar(ind+i*width, avgs[:,i], width, yerr=ebars[:,i], color=color, hatch=hatch))#, ecolor='r', capsize=6)color=lightblue,

  # add some
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_title(title)
  ax.set_xticks(ind+width/2)
  ax.set_xticklabels(xticklabels, rotation=30)

  if labels is not None:
    tmplist = [rects_i[0] for rects_i in rects_all]
    ax.legend(tuple(tmplist), tuple(labels), loc='lower right')
  # ax.legend( (rects1[0], rects2[0]), ('Men', 'Women') )

  def autolabel(rects):
      # attach some text labels
      for rect in rects:
          height = rect.get_height()
          ax.text(rect.get_x()+rect.get_width()/2., 1.01*height, '%.1f'%(height),
                  ha='center', va='bottom', fontsize=fontsize)

  for rects in rects_all:
    autolabel(rects)

  if bsavg is None or bsebar is None:
    print 'No baselines provided to plot'
  else:
    for i, val in enumerate([bsavg-bsebar, bsavg+bsebar]):
      lines = plt.axhline(y=val, linestyle='--')

    ax.text(lines.get_xdata()[0]+width, 1.01*bsavg, '%.1f'%(bsavg),
            ha='center', va='bottom', fontsize=fontsize)

  # plt.ylim([50,100])
  if aspect is not None:
    ax.set_aspect(aspect)
  plt.savefig(fname, dpi=300, bbox_inches='tight')
  plt.clf()


def plot_bar(avgs, yticklabels, xlabel, title, fname, rotation=0, yticklabelsize=None, aspect=None):
  # aspect can be 'auto' or 'equal'; use 'equal' for lengthy plots
  width = 0.35       # the width of the bars
  ind = width+np.arange(avgs.shape[0])  # the x locations for the groups

  all_colors = plt.rcParams['axes.color_cycle']
  all_colors.extend(plt.rcParams['axes.color_cycle'])
  fig, ax = plt.subplots()
  if aspect is not None:
    ax.set_aspect(aspect)
  if avgs.shape[1]==1:
    # pdb.set_trace()
    rects = ax.barh(ind, avgs[:,0], width)#, ecolor='r', capsize=6)color=lightblue,
  else:
    for i in range(avgs.shape[1]):
      rects1 = ax.barh(ind+i*width, avgs[:,i], width, color=all_colors[i])#, ecolor='r', capsize=6)color=lightblue,
    # rects2 = ax.bar(ind+width, avgs[:,1], width, )
  # add some
  # ax.set_aspect('auto')
  if aspect is not None:
    pdb.set_trace()
    ax.set_aspect(aspect)
  ax.set_xlabel(xlabel)
  ax.set_title(title)
  ax.set_yticks(ind+width/2)
  ax.set_yticklabels(yticklabels, rotation=rotation)
  if yticklabelsize is not None:
    ax.tick_params(axis='y', labelsize=yticklabelsize)
  plt.savefig(fname, dpi=300, bbox_inches='tight')


# Class to handle plotting
class plotter(object):

  def __init__(self, dataset, savefolder, debug=False):

    self.d = dataset.d                # pandas dataframe containing the data
    self.colattr = dataset.colattr
    self.savefolder = savefolder
    self.debug = debug
    self.groupby = 'tot'

    self.plotnum = 0
    self.plotdesc = []
    self.metadata = {}
    return None

  def get_prct_entropy(self, mulcounts):
    prct_entropy = 100.*scipy.stats.entropy(mulcounts)/data.get_maxentropy(len(mulcounts))
    if isnan(prct_entropy):
      mulcounts_nonan = self.dropNaNs_ifexist(mulcounts)
      if len(mulcounts_nonan)==0:
        return 50.
      prct_entropy = 100.*scipy.stats.entropy(mulcounts_nonan)/data.get_maxentropy(len(mulcounts_nonan))
    return prct_entropy

  def plotmuls(self, mulcounts, plotfun, numuniqvals, title, colnames, maxuniqs):
    prct_entropy = self.get_prct_entropy(mulcounts)
    if self.debug:
      title = '%s; En %.1f' % (title, prct_entropy)

    # if number of unique values is less than a threshold
    if numuniqvals<=maxuniqs:
      if prct_entropy<=MAXPERCT_ENTROPY:
        plotfun(mulcounts.values, mulcounts.index, title, colnames)

    # if number of unique values is too large, try clubbing lower counts together
    else:
      # simply do a bar plot on top-k elements after clubbing everything else together
      tmpmulcounts = mulcounts.iloc[:maxuniqs]
      clubbedcounts = 0.
      if len(colnames)==2: # if this is attempted with two columns, take the sum/avg of other records accordingly
        if self.groupby=='sum':
          clubbedcounts = sum(mulcounts.iloc[maxuniqs-1:])
        elif self.groupby=='avg':
          clubbedcounts = mulcounts.iloc[maxuniqs-1:].values.mean()
      elif len(colnames)==1: # if this is attempted with a single column, this is a pie chart, so take the sum of other records
        clubbedcounts = sum(mulcounts.iloc[maxuniqs-1:])

      tmpindex='others'
      if tmpindex in tmpmulcounts.index:
        tmpindex = self.generate_tmpindex(tmpindex, tmpmulcounts.index)
      tmpmulcounts = tmpmulcounts.append(pandas.Series(clubbedcounts, index=[tmpindex]))

      if prct_entropy<=MAXPERCT_TOPKENTROPY:
        if len(colnames)==1:
          title = 'Top-k records by %s' % (colnames[0])# , Oth %d' % #, sum(mulcounts[maxuniqs:])))
        elif len(colnames)==2:
          title = 'Top-k total %s by %s' % (colnames[0], colnames[1])# , Oth %d' % #, sum(mulcounts[maxuniqs:])))
        self.plot_bar(tmpmulcounts.values, tmpmulcounts.index, title, colnames)


  def plotcols(self):
    # single column plots
    for i in range(self.d.shape[1]):
      # if column is categorical
      if self.colattr['dtype'][i]=='mul':
        colcounts = self.colattr['valcounts'][i]
        self.plotmuls(colcounts, self.plot_pie, self.colattr['numuniqvals'][i], 'No. of Records by %s' % self.d.columns[i], [self.d.columns[i]], MAXUNIQS_FULLPIE)

      # if column is non-categorical
      else:
        try:
          self.plot_hist(self.d.ix[:,i], self.d.columns[i], 'Histogram', [self.d.columns[i]])
        except:
          type, value, tb = sys.exc_info()
          traceback.print_exc()
          print 'plot_hist charting failed!'

    # double column plots
    for i in range(self.d.shape[1]):
      for j in range(self.d.shape[1]):
        # if column-1 is categorical
        if self.colattr['dtype'][i]=='mul':
          # if column-2 is non-categorical
          if self.colattr['dtype'][j]!='mul':
            dtmp = self.d.ix[:,[i,j]]
            colname1 = self.d.columns[i]
            colname2 = self.d.columns[j]
            dtmp_sum = dtmp.groupby(colname1).sum().sort(colname2, ascending=False)[colname2]
            dtmp_mean = dtmp.groupby(colname1).mean().sort(colname2, ascending=False)[colname2]

            if self.groupby=='tot':
              self.plotmuls(dtmp_sum, self.plot_bar, self.colattr['numuniqvals'][i], 'Total %s by %s' % (colname2, colname1), [colname2, colname1], MAXUNIQS_FULLBAR)
            elif self.groupby=='avg':
              self.plotmuls(dtmp_mean, self.plot_bar, self.colattr['numuniqvals'][i], 'Average %s by %s' % (colname2, colname1), [colname2, colname1], MAXUNIQS_FULLBAR)

        # Plot scatter
        if self.colattr['dtype'][i]!='mul':
          if self.colattr['dtype'][j]!='mul':
            if i>j:
              continue
              tmpxy = self.d.ix[:,[i,j]]
              nbins = 50
              tmpxy = self.dropNaNs_ifexist(tmpxy)
              n = plt.hist2d(tmpxy.ix[:,0], tmpxy.ix[:,1], nbins)[0].ravel()
              plt.clf()
              prct_entropy = 100.*scipy.stats.entropy(n)/data.get_maxentropy(len(n))
              if prct_entropy < MAXPERCTENT_SCATTER:
                if self.debug:
                  title = '%s; En %.1f' % ('Histogram', prct_entropy)
                self.plot_scatter(tmpxy.ix[:,0], tmpxy.ix[:,1], self.d.columns[i], self.d.columns[j], title, [self.d.columns[i], self.d.columns[j]])

    self.metadata['colnames'] = self.d.columns.values.tolist()
    f = open('%s/plotdata.json' % (self.savefolder), 'wb')
    json.dump(self.metadata, f)
    f.close()
    return 0

  def dropNaNs_ifexist(self, df):
    if df.isnull().values.any():
      print 'WARNING: Data contains NaN values. Dropping these records..'
      df =  df.dropna(axis=0)
    return df

  def print_plot(self, plotname, colnames):
    print 'Plotting %d..' % self.plotnum
    plot_fname = u'plot_%d.png' % self.plotnum
    plot_path = '%s/%s' % (self.savefolder, plot_fname)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight') #
    self.plotnum += 1
    self.plotdesc.append(plotname)
    self.metadata[plot_fname] = colnames
    plt.clf()
    return 0

  def plot_pie(self, counts, labels, title, colnames):
    explode = [0]*len(counts)
    explode[0] = 0.1
    fig, ax = plt.subplots()
    plt.axis('equal')
    plt.pie(counts, labels=labels, autopct='%1.1f%%', explode=explode, startangle=90) # shadow=True
    plt.title(title)
    self.print_plot('pie-1', colnames)
    return 0

  def plot_hist(self, col, xlabel, title, colnames):
    col = self.dropNaNs_ifexist(col)
    nbins = 50
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(col, nbins)  # , facecolor='green', alpha=0.75 , normed=1
                                            # TODO possible extra feature -- when outliers exist, use range parameter
    prct_entropy = 100.*scipy.stats.entropy(n)/data.get_maxentropy(len(n))
    if self.debug:
      title = '%s; En %.1f' % (title, prct_entropy)
    if prct_entropy>MAXPERCT_ENTROPY:
      return 0
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Counts')
    ax.set_title(title)
    self.print_plot('hist-1', colnames)
    # ax.grid(True)
    return 0

  def plot_bar(self, avgs, labels, title, colnames):
    assert(len(avgs)==len(labels))
    width = 0.35                      # the width of the bars
    ind = width+np.arange(len(avgs))  # the x locations for the groups

    fig, ax = plt.subplots()
    rects = ax.barh(ind, avgs, width) #, ecolor='r', capsize=6)color=lightblue,

    # ax.set_xlabel('Feature Importances')
    ax.set_title(title)
    ax.set_yticks(ind+width/2)
    ax.set_yticklabels(labels)

    self.print_plot('bar-2', colnames)
    return 0

  def plot_scatter(self, x, y, xlabel, ylabel, title, colnames):
    fig, ax = plt.subplots()
    # scatter_contour(x, y, threshold=200, log_counts=True, ax=ax,
    #                 histogram2d_args=dict(bins=40),
    #                 plot_args=dict(marker=',', linestyle='none', color='black'),
    #                 contour_args=dict(cmap=plt.cm.bone))
    plt.scatter(x, y, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    self.print_plot('scatter-2', colnames)
    return 0

  def generate_tmpindex(self, basestr, index):
    tmpint = 0
    while 1:
      tmpindex_new = basestr + str(tmpint)
      if tmpindex_new in tmpcolcounts.index:
        tmpint += 1
      else:
        return tmpindex_new

  # For Debugging only: To understand how maximum entropy value changes with size of the discrete distribution
  def debugplot_entropywithN(self):
    x=range(2,40)
    y=[]
    for i in x:
      y.append(data.get_maxentropy(i))
    self.plot_bar(y,x,'Entropies', None)
    return 0

