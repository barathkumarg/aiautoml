from app import *
import sqlite3
import statistics
import time
import threading
filename = 'stats.json'

@app.route('/monitoring/system',methods=['POST','GET'])
def monitoring_system():
        return render_template('monitoring/system_monitoring.html')


@app.context_processor
def inject_load():
    while True:

        with open(filename, "r") as final:
            data = json.load(final)
            row_length = len(data['Total_Memory_(GB)'])

        return {'data':data,'row_length':row_length}

@app.before_first_request
def before_first_request():
    print('createdthrwead')
    threading.Thread(target=update_load).start()

def update_load():
    with app.app_context():
        while True:
            time.sleep(10)
            turbo.push(turbo.replace(render_template('monitoring/dynamic_contents.html'), 'load'))