import sys, glob
sys.path.append('gen-py')
sys.path.insert(0, glob.glob('/opt/thrift-0.9.3/lib/py/build/lib*')[0])

import os
import subprocess

from demo_dm import DM
from demo_dm.ttypes import *

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol


input_path = '../lstm'

try:
    transport = TSocket.TSocket('localhost', 19201)
    transport = TTransport.TBufferedTransport(transport)
    protocol = TBinaryProtocol.TBinaryProtocol(transport)
    client = DM.Client(protocol)

    transport.open()

    client.ping()
    #client.reset_ltm()
    fd = open('%s/sample_input.json' % input_path, 'r')
    dm_input = ""
    for line in fd:
        dm_input = dm_input + line
    print dm_input
    response = client.run(dm_input)
    print response
    transport.close()

except Thrift.TException as tx:
    print(('%s' % (tx.message)))


