#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# A collection of utility functions
#
import os, sys, csv, logging, json, math, datetime
import pandas, subprocess
import cPickle as pickle
from pytz import timezone
from pymongo import MongoClient
client = MongoClient() # get a client

const = 1./(1024**2)

# def get_mempid(pid):
#   from psutil.Process import get_memory_info
#   return get_memory_info(pid)

# def parallelize_evaluation():

def issorted_ascending(l):
  return all(l[i] <= l[i+1] for i in xrange(len(l)-1))

def rmfile(fname):
  try:
    os.remove(fname)
  except OSError:
    pass

def kill_proc_tree(pid, including_parent=True):
    parent = psutil.Process(pid)
    for child in parent.get_children(recursive=True):
        child.kill()
    if including_parent:
        parent.kill()

def check_pid(pid):
  """ Check For the existence of a unix pid. """
  return psutil.pid_exists(pid)
  # try:
  #   os.kill(pid, 0)
  # except OSError:
  #   return False
  # else:
  #   return True

def get_meminfo():
  from psutil import virtual_memory
  mem_snap = virtual_memory()
  mem_avail = mem_snap[1] * const
  mem_used = (mem_snap[0] - mem_snap[1])*const

  return {'avail_MB':mem_avail, 'used_MB':mem_used}


def read_jsonfile(fname):
  with open(fname) as infile:
    data = json.load(infile)
  return data

def write_jsonfile(data, fname):
  with open(fname, 'w') as outfile:
    json.dump(data, outfile)

def drop_duplicate_rows(dtmp):
  dtmp['tmpindex'] = dtmp.index
  dtmp.drop_duplicates(cols='tmpindex', take_last=True, inplace=True)
  del dtmp['tmpindex']
  return dtmp


def concat_csvs_cols_2ton(csvfiles, colnames, savefname=None):
  assert(len(csvfiles)==len(colnames))
  for fname in csvfiles:
    exit_no_file(fname)

  djoined = pandas.read_csv(csvfiles[0])
  djoined.columns = [djoined.columns[0], get_basename(csvfiles[0])]
  for fname in csvfiles[1:]:
    dtmp = pandas.read_csv(fname)
    dtmp.columns = [dtmp.columns[0], get_basename(fname)]
    assert(djoined.shape[0]==dtmp.shape[0])
    djoined = djoined.join(dtmp.ix[:,1:], how='inner')

  new_colnames = [djoined.columns[0]]
  new_colnames.extend(colnames)
  djoined.columns = new_colnames

  if savefname is None:
    savefname = 'resjoin'
    for fname in csvfiles:
      savefname += '--'+get_basename(fname)
    savefname = '/tmp/preds_%s.csv' % savefname
    # + get_basename(csvf1) + '--' + get_basename(csvf2)

  djoined.to_csv(savefname, index=False)

  return (djoined, savefname)


def merge_csv_byindex(csvfiles, index_col, newindex_name, how, drop_duplicates=False):
  for csvf in csvfiles:
    exit_no_file(csvf)

  dtmp = pandas.read_csv(csvfiles[0], index_col=index_col)
  if drop_duplicates:
    dtmp = drop_duplicate_rows(dtmp)
  print 'Using %s ' % csvfiles[0], 'with index ', dtmp.index.name, '..' # , dtmp.index.name, 'with shape ', dtmp.shape, '..'
  djoined = dtmp

  # debugging
  # index_sets = []
  for csvf in csvfiles[1:]:
    dtmp = pandas.read_csv(csvf, index_col=index_col)
    if drop_duplicates:
      dtmp = drop_duplicate_rows(dtmp)
    print 'Using %s ' % csvf, 'with index ', dtmp.index.name, '..' #, dtmp.index.name, 'with shape ', dtmp.shape, '..', djoined.shape
    # debugging
    # pdb.set_trace()
    # indset = set()
    # for ind in dtmp.index:
    #   indset.add(ind)
    # index_sets.append(indset)

    djoined = djoined.join(dtmp, how=how)
    djoined.index.name = newindex_name
    if drop_duplicates:
      djoined = drop_duplicate_rows(djoined)

  # debugging
  # for is1 in index_sets:
  #   for is2 in index_sets:
  #     print len(is1), len(is2), is1==is2

  print 'Joined %d files together..' % len(csvfiles)
  return djoined

def get_filesize_MB(fname):
  return os.path.getsize(fname)/(1024.*1024)

def read_pickle_ifexists(fname, disable_stdout=False):
  if os.path.isfile(fname):
    if not disable_stdout:
      print 'File %s already exists.. Reading %.1f MB from disk..' % (fname, get_filesize_MB(fname))
      sys.stdout.flush()
    return pickle.load(open(fname, 'rb'))
  else:
    return None

def drop_mongo(collection):
  client.db[collection].drop()

def write_pickle(obj, fname):
  pickle.dump(obj, open(fname, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

def write_mongo(collection, doc):
  coll = client.db[collection] # get the database, like a table in sql
  coll.insert(doc) # like insert a row in sql

def update_mongo(collection, doc):
  coll = client.db[collection]
  coll.update({'_id': doc['_id']}, doc, upsert=True)

def read_mongo(collection, querydoc):
  coll = client.db[collection]
  doc = coll.find_one(querydoc) # like find a row by particular column value in sql
  return doc

def read_mongo_itr(collection):
  coll = client.db[collection]
  return coll.find()

def rename_ifexists(fname):
  if os.path.isfile(fname):
    print 'File %s of size %.1f already exists.. Auto-renaming..' % (fname, get_filesize_MB(fname))
    sys.stdout.flush()
    last_mod_datetime = datetime.datetime.fromtimestamp(os.path.getmtime(fname))
    newfname = '%s--%s%s' % (os.path.splitext(fname)[0], last_mod_datetime.strftime('%Y-%m-%d--%H-%M-%S--z%Z%z'), os.path.splitext(fname)[1])
    os.rename(fname, newfname)
    print 'Renamed %s to %s..' % (fname, newfname)
    return 0
  else:
    return 0

def get_basename_os(fname):
  # If fname is of form '../../results/r5/oidcid_131001-140501.csv', then this returns 'oidcid_131001-140501.csv'
  return os.path.basename(fname)

def get_basename_os_without_extension(fname):
  # If fname is of form '../../results/r5/oidcid_131001-140501.csv', then this returns 'oidcid_131001-140501'
  return os.path.splitext(os.path.basename(fname))[0]

def get_dirname_os(fname):
  # If fname is of form '../../results/r5/oidcid_131001-140501.csv', then this returns '../../results/r5'
  return os.path.dirname(fname)

def get_basename_expt(fname):
  # If fname is of form '../../results/r5/oidcid_131001-140501.csv', then this returns '131001-140501'
  # Please note that the fname should contain an extension (of any length), and that the base name of the fname should
  # contain exactly one '_'
  basename = get_basename_os_without_extension(fname)
  return basename.split("_")[-1]

def get_fileextension(fname):
  # If fname is of form '../../results/r5/oidcid_131001-140501.csv', then this returns '.csv'
  return os.path.splitext(fname)[1]

def get_filepath_without_extension(fname):
  # If fname is of form '../../results/r5/oidcid_131001-140501.csv', then this returns '../../results/r5/oidcid_131001-140501'
  return os.path.splitext(fname)[0]

def ensuredir(dir):
  if not os.path.exists(dir):
      os.makedirs(dir)

def exit_if_file_present(fname):
  if os.path.isfile(fname):
    print 'File already exists: %s Quitting..' % fname
    sys.exit()

def exit_no_file(fname):
  if not os.path.isfile(fname):
    print 'File does not exist: %s Quitting..' % fname
    sys.exit()

def get_utc(locdate, loczone_str):
  loczone = timezone(loczone_str)
  loc_dt = loczone.localize(locdate)
  utczone = timezone('UTC')
  utc_dt = loc_dt.astimezone(utczone)
  # fmt = '%Y-%m-%d %H:%M:%S %Z%z'
  # print loc_dt.strftime(fmt)
  # utc_dt.strftime(fmt)
  return utc_dt

def get_sortedindex(lst, descending=False):
  sortind = range(len(lst))
  sortind.sort(key=lambda k: lst[k], reverse=descending)
  return sortind

def apply_sortedindex(lst, sortind):
  lst = [lst[i] for i in sortind]
  return lst

def get_header_dict(row):
  vals = range(len(row))
  return dict(zip(row,vals))

def get_numcolumns_csv(csvfname):
  # returns integer representing number of columns in csv file
  with open(csvfname, 'r') as f:
    reader=csv.reader(f,delimiter=',', quotechar='|')
    for row in reader:
      break
  return len(row)

def call_subproc(cmd):
  print 'doing $%s' % cmd
  subprocess.call(cmd, shell=True, executable='/bin/bash')


class Logger(object):
    logger_name = ''
    log_file = ''
    l = None
    def __init__(self, logger_name, log_file):
        self.logger_name = logger_name
        self.log_file = log_file

        self.l = logging.getLogger(logger_name)
        fileHandler = logging.FileHandler(log_file, mode='w')

        self.l.setLevel(logging.DEBUG)
        self.l.addHandler(fileHandler)


    def log(self, **kwargs):
        logn = logging.getLogger(self.logger_name)
        logn.info(json.dumps(kwargs))

    def logprint(self, **kwargs):
        lkwargs = list((k,v) for (k,v) in kwargs.iteritems())
        lkwargs.sort(key=lambda tup: tup[0])
        for (k,v) in lkwargs:
            num_tabs = int(math.ceil( (len(k) + 1 + len(': ')) / 8 ))
            tabs = ''.join(['\t' for i in range(4-num_tabs)])
            print k, ': %s' % tabs, v
        self.log(**kwargs)



class runtimetracker(object):

  def __init__(self, reporting=True):
    self.reporting = reporting
    self.start = None

  def tick(self, message=''):
    self.start = datetime.datetime.now()
    if self.reporting:
      self.tick_message = message
      if len(message)>0:
        print '%s.. ' % message,
        sys.stdout.flush()

  def tock_sec(self):
    return (datetime.datetime.now() - self.start).seconds

  def tock(self, message=''):
    if self.reporting:
      runtime = datetime.datetime.now() - self.start
      runtime_str = str(runtime).split('.', 2)[0]
      if len(message)==0:
        if len(self.tick_message)==0:
          print 'took(%s)'%(runtime_str)
        else:
          print '(%s)took(%s)' % (self.tick_message, runtime_str)
      else:
        print '(%s)took(%s)' % (message, runtime_str)
      sys.stdout.flush()
    self.start = None

  def tocktick(self, message=''):
    if self.start is not None:
      self.tock()
    self.tick(message)
