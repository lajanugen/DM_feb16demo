#!/usr/bin/env python

import sys, glob
sys.path.append('../../gen-py')
sys.path.insert(0, glob.glob('/opt/thrift-0.9.2/lib/py/build/lib.*')[0])

import dm_service
from ttypes import *

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

try:

  # Make socket
  transport = TSocket.TSocket('localhost', 9890)

  # Buffering is critical. Raw sockets are very slow
  transport = TTransport.TBufferedTransport(transport)

  # Wrap in a protocol
  protocol = TBinaryProtocol.TBinaryProtocol(transport)

  # Create a client to use the protocol encoder
  client = dm_service.Client(protocol)

  # Connect!
  transport.open()

  output = client.act('hello')
  print "in client"
  print '%s' % (output)

  # Close!
  transport.close()

except Thrift.TException, tx:
  print '%s' % (tx.message)
