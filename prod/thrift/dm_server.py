import sys, glob
sys.path.append('gen-py')
sys.path.insert(0, glob.glob('/opt/thrift-0.9.3/lib/py/build/lib*')[0])

import os
import subprocess

from demo_dm import DM
from demo_dm.ttypes import *

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

executable_path = '../lstm'
executable = 'serve_dm.lua'

class DMServerHandler:
    def __init__(self):
        self.log = {}

    def ping(self):
        print('ping()')

    def run(self, dm_input):
        owd = os.getcwd()
        global executable_path
        os.chdir(executable_path)

        #os.system('rm -f db.json')
        cmd = ['th', '%s'%executable]
        print ('Received a new query: %s'%dm_input)
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        out = p.communicate(input='%s'%dm_input)[0]
        print 'Query served: %s'%out
        os.chdir(owd)
        return out

    def reset_ltm(self):
        owd = os.getcwd()
        global executable_path
        os.chdir(executable_path)
        os.system('rm -f db.json')
        os.chdir(owd)


handler = DMServerHandler()
processor = DM.Processor(handler)
transport = TSocket.TServerSocket(port=19201)
tfactory = TTransport.TBufferedTransportFactory()
pfactory = TBinaryProtocol.TBinaryProtocolFactory()

server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)

print('Starting the server')
server.serve()
print('done.')

    
    
