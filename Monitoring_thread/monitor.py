import psutil
from datetime import datetime
import threading
import os
import json
import time
import sqlite3
filename= 'stats.json'


class Monitor:

    def __init__(self):

        self.key = threading.Event()
        self.key.set()  # set the key for loop execution


    def __call__(self,param_args):
        def wrapper(*args):
            self.start_time = time.time()
            self.create_json()
            monitor_thread = threading.Thread(target= self.__fetch_details,args=(os.getpid(),),daemon=True,name='monitor_thread')
            monitor_thread.start()
            param_args(*args)
            self.key.clear()
            monitor_thread.join()

        return wrapper
    def create_json(self):
        with open(filename, 'w') as f:
            data = {
                'Total_Memory_(GB)' : [],'Avaliable_Memory_(GB)': [],'Used_Memory_(GB)': [],
                'Memory_Percent_(%)': [],'CPU_Percent_(%)': [],'Process_Memory_(GB)': [],
                'Process_CPU_(%)': [],'Process_threads' : [],'Process_read_bytes' : [],'Process_write_bytes' : [],'x_axis':[],
                'Process_id' : os.getpid(),'Up_time': self.get_uptime()
            }
            json.dump(data, f, indent=4)


    def write_json(self, data):
        """Write json file with provided data"""
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

    def append(self,rows):
        with open("stats.json", "r") as final:
            data = json.load(final)
            pos = 0
            for key in data:
                if key == 'Process_id' or key == 'Up_time':
                    pass
                else:
                    data[key].append(rows[pos])
                    pos+=1
            self.write_json(data)

    def read(self):
        with open(filename, "r") as final:
            info_data = json.load(final)
            print(info_data)

    def __fetch_details(self,pid):

        print('connectio established')
        while self.key.is_set():
            result_set = []
            #--------------- System details -------------------- #
            result_set.append(psutil.virtual_memory().total / 1024 ** 3)
            result_set.append(psutil.virtual_memory().available / 1024 ** 3)
            result_set.append(psutil.virtual_memory().used / 1024 ** 3)
            result_set.append(psutil.virtual_memory().percent)
            result_set.append(psutil.cpu_percent())

            #--------------- Process details ---------------------#
            p = psutil.Process(pid)
            result_set.append(p.memory_info().rss / 1024 ** 3)
            result_set.append(p.cpu_percent())
            result_set.append(p.num_threads())
            result_set.append(p.io_counters().read_bytes)
            result_set.append(p.io_counters().write_bytes)
            result_set.append(round((time.time() - self.start_time),1))

            #------------------ push to json ----------------------------#
            self.append(result_set)
            # self.read()
            time.sleep(7) #push logs to the json



    def get_uptime(self):
        now = datetime.now()
        return now.strftime("%d/%m/%Y %H:%M:%S")

    def get_time(self):
        now = datetime.now()
        return now.strftime("%H:%M:%S")






