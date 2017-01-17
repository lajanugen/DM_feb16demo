#!/usr/bin/env python

import sys, glob
sys.path.append('../../gen-py')
sys.path.insert(0, glob.glob('/opt/thrift-0.9.2/lib/py/build/lib.*')[0])

import dm_service
from ttypes import *

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

import socket

class dm_service_handler:
  def __init__(self):
    self.log = {}

  def act(self, text):
    print "in server.act()"
    # Example text:<class:id=492,department=EECS,semester=2030,instructor=unknown>
    # Example text:<class:id=492,department=EECS,credits=unknown>
    
    entity_str = text.rstrip('>').lstrip('<')
     
    entity_name = entity_str.split(':')[0]
    sv_pairs = entity_str.split(':')[1].split(',')
    
    sv_pair_map = {sv.split('=')[0]: sv.split('=')[1] for sv in sv_pairs}

    response = ""
    for k, v in sv_pair_map.items():
        print k, v
        if v == 'unknown':
            if k == 'credits':
                response = "%s-%s is a 3 credit course. " % (sv_pair_map['department'], sv_pair_map['id'])
            elif k == 'instructor':
                response += "The instructor of {department}-{id} is {instructor}. ".format(department=sv_pair_map['department'],
                                                                                           id=sv_pair_map['id'],
                                                                                           instructor="Professor Benjamin Kuipers"
                                                                                          )
            elif k == 'description':
                response += "{department}-{id} is an introductory course to artificial intelligence. The purpose of this course is to provide an overview of this field. We will cover topics including: agents, search, planning, uncertainty, and learning. The goals of this course are to provide a fundamental knowledge of the field. The course evaluation will include homework, a midterm, a final, and in-class quizzes. ".format(department=sv_pair_map['department'], id=sv_pair_map['id'])
            elif k == 'prerequisites':
                response += "{department}-{id} requires that students should have taken EECS 281 and/or have graduate standing. ".format(department=sv_pair_map['department'], id=sv_pair_map['id'])

            else:
                response += "Sorry, I don't understand your question. Please ask again."
            #response = "Do you want to know more about {attr} of this {target}?".format(attr=k, target=entity_name);
    # NLG service (text)
#    
    return response


handler = dm_service_handler()
processor = dm_service.Processor(handler)
transport = TSocket.TServerSocket(port=5999)
tfactory = TTransport.TBufferedTransportFactory()
pfactory = TBinaryProtocol.TBinaryProtocolFactory()

server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)

# You could do one of these for a multithreaded server
#server = TServer.TThreadedServer(processor, transport, tfactory, pfactory)
#server = TServer.TThreadPoolServer(processor, transport, tfactory, pfactory)

print "Starting python server..."
server.serve()
print "done!"
