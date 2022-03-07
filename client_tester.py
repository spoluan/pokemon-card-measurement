# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:17:34 2022

@author: Sevendi Eldrige Rifki Poluan
""" 
 
import socket  
import sys
 
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print ("Socket successfully created")
except socket.error as err:
    print ("socket creation failed with error %s" %(err))
  
port = 20220
 
try:
    print('Resolving host name ...')
    host_ip = socket.gethostbyname('127.0.0.1')
    print('Resolved')
except socket.gaierror: 
    print("there was an error resolving the host")
    sys.exit() 
    
print('Connecting ...')
s.connect((host_ip, port))
print('Connected')

print('Sending command ...')
status = 'STOP SERVER' # RUN, FORCE STOP, RESULTS, CURRENT, STOP SERVER
s.send(status.encode());

print (s.recv(1024).decode())

s.close()

