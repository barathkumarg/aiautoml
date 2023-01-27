# import sqlite3
#
# conn = sqlite3.connect('stats.db')
#
# cur = conn.cursor()
# cur.execute("SELECT * FROM Numerical_stats")
# rows = cur.fetchall()
#
# # ---------------------------- list logic ---------------------#
# sys_memory_total = list()
# sys_memory_available = list()
# sys_memory_used = list()
# sys_memory_percent = list()
# sys_cpu_percent = list()
# proc_memory_used = list()
# proc_cpu_percent = list()
# proc_threads = list()
# proc_id = list()
# proc_read_bytes = list()
# proc_write_bytes = list()
#
# for row in rows:
#     sys_memory_total.append(round(row[0], 2))
#     sys_memory_available.append(round(row[1], 2))
#     sys_memory_used.append(round(row[2], 2))
#     sys_memory_percent.append(round(row[3], 2))
#     sys_cpu_percent.append(round(row[4], 2))
#     proc_memory_used.append(round(row[5], 2))
#     proc_cpu_percent.append(round(row[6], 2))
#     proc_threads.append(round(row[7], 2))
#     proc_id.append(round(row[8], 2))
#     proc_read_bytes.append(round(row[9], 2))
#     proc_write_bytes.append(round(row[10], 2))
#
# dict = {'sys_memory_total':sys_memory_total,'sys_memory_available':sys_memory_available,'sys_memory_used':sys_memory_used,
#         'sys_memory_percent':sys_memory_percent,'sys_cpu_percent':sys_cpu_percent,'proc_memory_used':proc_memory_used,
#         'proc_threads':proc_threads,'proc_id':proc_id,'proc_read_bytes':proc_read_bytes,'proc_write_bytes':proc_write_bytes}
# print(dict)
# for key,value in dict.items():
#     print(dict[key][-1])
import json
filename = 'stats.json'
# list of dictionaries of employee data
data = {"id": ["1", "2", "3"], "name": ["bhanu", "sivanagulu"],

 "department": ["HR", "IT"]}

def write_json(data):
    """Write json file with provided data"""
    with open(filename,'w') as f:
        json.dump(data, f, indent=4)

def append():
    with open("stats.json", "r") as final:
        # json.dump(data, final)
        info_data = json.load(final)
        print(info_data.keys())
        info_data["id"].append(4)
        write_json(info_data)

def read():
    with open(filename, "r") as final:
        info_data = json.load(final)
        print(info_data)

read()
append()
read()

# display
