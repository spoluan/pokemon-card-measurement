# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:17:34 2022

@author: Sevendi Eldrige Rifki Poluan
""" 
 
import socket  
import sys
import time
 
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

status = 'RUN\r' # RUN\r, FORCE STOP\r, RESULTS\r, CURRENT\r, STOP SERVER\r
  
# For loop is made for the sake of testing only to check the robustness of the server
for x in range(1):
    s.send(status.encode())
    rec = s.recv(1024).decode()
    print(rec)
    time.sleep(1)
    
status = 'CURRENT\r' # RUN\r, FORCE STOP\r, RESULTS\r, CURRENT\r, STOP SERVER\r
  
# For loop is made for the sake of testing only to check the robustness of the server
for x in range(1):
    s.send(status.encode())
    rec = s.recv(1024).decode()
    print(rec)
    time.sleep(1)
    
    
status = 'FORCE STOP\r' # RUN\r, FORCE STOP\r, RESULTS\r, CURRENT\r, STOP SERVER\r
  
# For loop is made for the sake of testing only to check the robustness of the server
for x in range(1):
    s.send(status.encode())
    rec = s.recv(1024).decode()
    print(rec)
    time.sleep(1)
    

status = 'STOP SERVER\r' # RUN\r, FORCE STOP\r, RESULTS\r, CURRENT\r, STOP SERVER\r
  
# For loop is made for the sake of testing only to check the robustness of the server
for x in range(1):
    s.send(status.encode())
    rec = s.recv(1024).decode()
    print(rec)
    time.sleep(1)

# Server can receive command without closing
s.close() 
 

