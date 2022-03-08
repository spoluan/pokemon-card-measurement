# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:20:39 2022

@author: Sevendi Eldrige Rifki Poluan
""" 
 
import socket  
import os  
import subprocess       
import threading # thread6 
import time 

class CardServer(object):
    
    def __init__(self):
        
        abs_path = os.path.dirname(__file__)
        self.addr = os.path.join(abs_path, 'sources' )
        self.addr_to_save = os.path.join(abs_path, 'outputs')
        self.results_path = os.path.join(abs_path, 'results')
        self.main_path = abs_path
        self.ip = '127.0.0.1'
        self.port = 20220
        self.is_running = False

    def read_write_information(self, path, file, write=False):
        status = ''
        used = False
        while not used:
            try: 
                with open(os.path.join(path, file), 'r+') as rw:  
                    status = rw.read().strip() 
                    if write == True: 
                        rw.seek(0)
                        rw.write('')
                        rw.truncate()
                used = True
            except Exception as err: 
                print('ISSUES:', err) 
                
        return status
    
    def write_information(self, path, file, contents=''):
        used = False 
        while not used:
            try:
                with open(os.path.join(path, file), 'w') as w:  
                    w.write(contents) 
                used = True
            except Exception as err: 
                print('ISSUES:', err) 
                
        return 'DONE'
    
    def start_app(self, main_path):
        try:
            cmd = f'python {os.path.join(main_path, "app.py")}' 
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            out, err = p.communicate()  
            # output = p.stdout.readline() 
        except Exception as a:
            print('ISSUES:', a)
             
    def recv_data(self, c):  
        
        is_stop_server = False
        while True:  
            
            try:
                recv = c.recv(1024).decode() 
            except Exception as a:
                print('ISSUES:', a)   
                
            if recv != '': 
                
                if recv.strip() == 'RUN': 
                    print('RUN')
                    if not self.is_running:
                        a_thread = threading.Thread(target=self.start_app, args = (self.main_path, ))
                        a_thread.start()
                        c.send('APP HAS STARTED'.encode()) 
                        self.is_running = True
                    else:
                        c.send('APP IS RUNNING'.encode()) 
                    print('DONE')
                    
                elif recv.strip() == 'FORCE STOP':
                    print('>> FORCE STOP ...')
                    if self.write_information(self.results_path, 'running_status.txt', contents='FORCE STOP') == 'DONE':
                        c.send('DONE FORCE STOP'.encode()) 
                    print('DONE')
                    self.is_running = False
                    
                elif recv.strip() == 'RESULTS':
                    print('>> RESULTS ...')
                    results = self.read_write_information(self.results_path, 'results.txt', write=True).encode()
                    if len(results.strip()) > 0:
                        c.send(results)
                    else:
                        c.send(b'EMPTY')
                    print('DONE')
                    
                elif recv.strip() == 'CURRENT':
                    print('>> CURRENT ...')
                    current = self.read_write_information(self.results_path, 'current_process.txt', write=False).encode()
                    if len(current.strip()) > 0:
                        c.send(current)
                    else:
                        c.send(b'EMPTY')
                    print('DONE')
                    
                elif recv.strip() == 'STOP SERVER': 
                    print('>> STOP SERVER ...') 
                    stop_server = self.write_information(self.results_path, 'running_status.txt', contents='FORCE STOP')
                    self.is_running = False 
                    is_stop_server = True 
                    if stop_server == 'DONE':
                        c.send('SERVER_STOPPED'.encode())   
                    print('DONE')
                    break
                
                else:
                    c.send('UNKNOWN COMMAND'.encode())
                    print('>> UNKNOWN COMMAND ...') 
                
                time.sleep(1)
            
            else: 
                
                break
            
        return is_stop_server
          
    def start_server(self):
      
        s = socket.socket()        
             
        s.bind((self.ip, self.port))        
        print ("SOCKET BIND TO %s" % (self.port))
             
        while True: 
            
            print('>> WAITING FOR REQUEST')
            
            s.listen(5)    
            print ("SOCKET IS LISTENING ...")
            
            c, addr = s.accept() # Command: RUN, FORCE STOP, RESULTS, CURRENT   
            print ('GOT A NEW CONNECTION FROM:', addr)
             
            if not self.recv_data(c): 
                continue 
            
            else:
                c.close()
                break 

if __name__ == '__main__':
    app = CardServer()
    app.start_server()
