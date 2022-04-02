# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 22:33:02 2022

@author: Sevendi Eldrige Rifki Poluan
"""

import os

class FileUtils(object): 

    def update_information(self, path, file, status):
        con = False
        while not con:
            try:
                with open(os.path.join(path, file), 'w') as w:  
                    w.write(f'{status}')
                con = True
            except:
                pass
            
    def write_results(self, append, path, skipped=False, write_status='w'):  
        if not skipped:  
            con = False
            while not con:
                try: 
                    with open(os.path.join(path, 'results.txt'), write_status) as w: 
                        a = ',' . join(list(map(lambda x: str(x), append))) + '\n'
                        if len(a.strip()) > 0:
                            w.write(a) 
                    con = True
                except:
                    pass
        else:
            con = False
            while not con:
                try:
                    with open(os.path.join(path, 'skipped_imgs.txt'), write_status) as w: 
                        a = ',' . join(list(map(lambda x: str(x), append))) + '\n'
                        if len(a.strip()) > 0:
                            w.write(a) 
                    con = True
                except:
                    pass 
            
    def is_to_stop(self, path, file):
        con = False
        status = ''
        while not con:
            try:
                with open(os.path.join(path, file), 'r') as w:  
                    status = w.read().strip()
                con = True
            except:
                pass 
        
        print('CURRENT STATUS: ', status)
        if status == 'FORCE STOP':
            return True
        else:
            return False
        
    def prepare_output_dir(self, folder_name):   
        [os.system(f'erase /s /q "{os.path.join(folder_name, x)}"') for x in os.listdir(os.path.join(os.path.realpath(__file__).replace(os.path.basename(__file__), ''), folder_name))]