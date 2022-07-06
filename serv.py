# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 11:16:39 2022

@author: Sevendi Eldrige Rifki Poluan
""" 
 
import socket  
import os  
import subprocess       
import threading # thread6 
import time 
import os 

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
        self.is_stop_server = False
        self.connected_client = []

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
                print('ISSUES (READ/WRITE):', err) 
                
        return status 
    
    def write_information(self, path, file, contents=''):
        used = False 
        while not used:
            try:
                with open(os.path.join(path, file), 'w') as w:  
                    print('Write', contents)
                    w.write(contents) 
                used = True
            except Exception as err: 
                print('ISSUES (WRITE):', err) 
                
        return 'DONE'
    
    def start_app(self, main_path):
        try:
            cmd = f'python {os.path.join(main_path, "app.py")}' 
            print('Run: ', cmd)
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            out, err = p.communicate()
            output = p.stdout.readline()  
        except Exception as a:
            print('ISSUES (START APP):', a)
            os._exit(0)

        print('START APP FINISH')
             
    def recv_data(self, c, addr, s):  
         
        while True:  
            
            if self.is_stop_server:
                s.close()
                break

            print('Now is listening command from', addr, self.is_stop_server)

            recv = ''
            try:
                recv = c.recv(1024).decode()
            except Exception as a:
                print('ISSUES:', a)
                self.is_stop_server = True 
                
            if recv != '': 
                
                print('Received', recv.strip())
                if recv.strip() == 'RUN': 
                    print('RUN')
                    if not self.is_running:
                        a_thread = threading.Thread(target=self.start_app, args = (self.main_path, ))
                        a_thread.start()
                        print('Send response: APP HAS STARTED')
                        c.send('APP HAS STARTED'.encode()) 
                        self.is_running = True
                    else:
                        print('Send response: APP IS RUNNING')
                        c.send('APP IS RUNNING'.encode()) 
                    print('DONE RUN')
                    
                elif recv.strip() == 'FORCE STOP':
                    print('>> FORCE STOP ...')
                    if self.write_information(self.results_path, 'running_status.txt', contents='FORCE STOP') == 'DONE':
                        c.send('DONE FORCE STOP'.encode()) 

                    print('DONE')
                    self.is_running = False
                    self.is_stop_server = True 

                    s.close()
                    
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
                    is_server_stop = self.read_write_information(self.results_path, 'running_status.txt', write=False).encode()
                    if is_server_stop != 'STOP':
                        current = self.read_write_information(self.results_path, 'current_process.txt', write=False).encode()
                        if len(current.strip()) > 0:
                            c.send(current)
                        else:
                            c.send(b'EMPTY')
                        print('DONE')
                    else:
                        c.send(b'STOP')
                        print('DONE')

                elif recv.strip() == 'SEND_STOP':
                    for x in self.connected_client:
                        x.send(b'STOP')
                    
                elif recv.strip() == 'STOP SERVER': 
                    print('>> STOP SERVER ...') 
                    stop_server = self.write_information(self.results_path, 'running_status.txt', contents='FORCE STOP')
                    
                    self.is_stop_server = True 
                    if stop_server == 'DONE':
                        c.send('SERVER_STOPPED'.encode())   
                    print('DONE')
                    
                    s.close()
                    break
                
                else:
                    c.send('UNKNOWN COMMAND'.encode())
                    print('>> UNKNOWN COMMAND ...') 
                
                time.sleep(1)

                print('DONE ALL')
            else:  
                break  
            
        print('Thread finish!')  

    def real_time_notification(self, c, addr, s):
        current_temp = ''
        final_card = ''
        counter = 0 
        while True: 

            current = self.read_write_information(self.results_path, 'current_process.txt', write=False) 
         
            if current != current_temp.strip():
                current_temp = current 
                try:
                    s = list(map(lambda x: int(x), current.split('>')[1].split("from")))
                    print(s)
                    if(s[0] == s[1]):
                        final_card = current.split('>')[0]
                        print('FINAL CARD', final_card)
                    counter += 1
                except:
                    print('Error')
                c.send(f'CURRENT + {current}' . encode()) 

            results = self.read_write_information(self.results_path, 'results.txt', write=True)

            if len(results.strip()) > 0: 
                if final_card.split('.')[0] == str(results.strip()).split('.')[0]: 
                    c.send(f'RESULT + {results} + STOP' . encode())  
                else:    
                    c.send(f'RESULT + {results}' . encode())
              
        
    def start_server(self):
      
        s = socket.socket()        
             
        s.bind((self.ip, self.port))        
        print ("SOCKET BIND TO %s" % (self.port))
             
        while True: 
            
            print('>> WAITING FOR REQUEST')
            
            s.listen(5)     
            
            try:
                c, addr = s.accept() # Command: RUN, FORCE STOP, RESULTS, CURRENT   
                print ('GOT A NEW CONNECTION FROM:', addr)

                self.connected_client.append(c)
                
                # Create thread for data receiving
                threading.Thread(target=self.recv_data, args=(c, addr, s, )).start()  

                threading.Thread(target=self.real_time_notification, args=(c, addr, s, )).start()  
 
            except:
                print('Server is forced stop by the client!')
            
            print('SERVER STATUS', self.is_stop_server)
            if not self.is_stop_server: 
                print('Continue')
                continue  
            else:
                s.close()
                print('Server closed')
                break 

        print('DONE SERVER')

if __name__ == '__main__':
    app = CardServer()
    app.start_server()