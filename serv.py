# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:20:39 2022

@author: Sevendi Eldrige Rifki Poluan
""" 
 
import socket  
import os  
import subprocess       
import threading # thread6 

class CardServer(object):
    
    def __init__(self):
        
        self.addr = './sources' 
        self.addr_to_save = './outputs'
        self.results_path = './results' 
        self.main_path = "."
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
                        print('Write ..', write)
                        rw.seek(0)
                        rw.write('')
                        rw.truncate()
                used = True
            except Exception as err: 
                print(err)
        print('Result to return', status)
        return status
    
    def write_information(self, path, file, contents=''):
        used = False
        while not used:
            try:
                with open(os.path.join(path, file), 'w') as w:  
                    print('Is file exist:', os.path.isfile(os.path.join(path, file)))
                    w.write(contents)
                    print('Finished writing ...')
                used = True
            except Exception as err: 
                print(err)
        return 'DONE'
    
    def start_app(self, main_path):
        try:
            cmd = f'python {os.path.join(main_path, "app.py")}' 
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            out, err = p.communicate()  
            output = p.stdout.readline()
            print(output, out, err)
        except Exception as a:
            print(a)
          
    def start_server(self):
      
        s = socket.socket()        
             
        s.bind((self.ip, self.port))        
        print ("socket binded to %s" %(self.port))
          
        s.listen(5)    
        print ("socket is listening")
        
        status = ''
        while status != 'STOP': 
            
            print('Waiting for request ...')
            
            c, addr = s.accept() # Command: RUN, FORCE STOP, RESULTS, CURRENT   
            print ('Got connection from', addr)
            
            # Handle data sent from client
            recv = c.recv(1024).decode()
            
            if recv.strip() == 'RUN': 
                if not self.is_running:
                    a_thread = threading.Thread(target=self.start_app, args = (self.main_path, ))
                    a_thread.start()
                    c.send('RUNNING'.encode())
                    print('RUN')
                    self.is_running = True
            elif recv.strip() == 'FORCE STOP':
                print('Force stop ...')
                if self.write_information(self.results_path, 'running_status.txt', contents='FORCE STOP') == 'DONE':
                    c.send('DONE FORCE STOP'.encode())
                print('Done')
                self.is_running = False
            elif recv.strip() == 'RESULTS':
                print('Get result ...')
                c.send(self.read_write_information(self.results_path, 'results.txt', write=True).encode())
                print('Done')
            elif recv.strip() == 'CURRENT':
                print('Get current running status ...')
                c.send(self.read_write_information(self.results_path, 'current_process.txt', write=False).encode())
                print('Done')
            elif recv.strip() == 'STOP SERVER':
                print('STOP SERVER')
                print('Force stop ...')
                if self.write_information(self.results_path, 'running_status.txt', contents='FORCE STOP') == 'DONE':
                    c.send('DONE FORCE STOP'.encode())
                c.close()
                status = 'STOP'
                self.is_running = False
            else:
                print('Unknown command')
            
            if recv.strip() != 'STOP SERVER':
                send = 'Connected'
                c.send(send.encode()) 

if __name__ == '__main__':
    app = CardServer()
    app.start_server()
