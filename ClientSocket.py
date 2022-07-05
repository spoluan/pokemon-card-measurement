# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 13:57:09 2022

@author: SEVENDI ELDRIGE RIFKI POLUAN
"""

import socket  
import sys 
import time

class Client(object):

    def __init__(self):
        try:
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print ("Socket successfully created")
        except socket.error as err:
            print ("socket creation failed with error %s" %(err))
        
        self.port = 20220
    
        try:
            print('Resolving host name ...')
            self.host_ip = socket.gethostbyname('127.0.0.1')
            print('Resolved')
        except socket.gaierror: 
            print("there was an error resolving the host")
            sys.exit() 

    def connect(self):
        print('Connecting ...')
        self.s.connect((self.host_ip, self.port))
        print('Connected')
    
    def force_stop(self):
        time.sleep(3)
        status = 'FORCE STOP\r' 
        self.s.send(status.encode())
        rec = self.s.recv(1024).decode()
        print(rec)
        time.sleep(1)
        