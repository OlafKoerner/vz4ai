import socket
from decouple import Config, RepositoryEnv, Csv
import logging
from logging.handlers import RotatingFileHandler
from gevent import monkey; monkey.patch_all() # https://bottlepy.org/docs/dev/async.html
import bottle
import pymysql
from bottle import response, post, hook
#from bottle_cors_plugin import cors_plugin
import json
import csv
import datetime
import shutil
import numpy as np # https://www.nbshare.io/notebook/505221353/ERROR-Could-not-find-a-version-that-satisfies-the-requirement-numpy==1-22-3/
import tensorflow as tf # https://qengineering.eu/install-tensorflow-2.2.0-on-raspberry-pi-4.html
import tensorflow.keras as keras

#global settings
config = Config(RepositoryEnv("./.env"))

#setup logger for file and console
#https://blog.sentry.io/logging-in-python-a-developers-guide/
logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s %(levelname)s %(message)s", 
        handlers=[RotatingFileHandler(filename="./restapi_bottle.log", maxBytes=300000, backupCount=2)])
logging.getLogger().addHandler(logging.StreamHandler())

app = bottle.Bottle()
#app.install(cors_plugin('*'))

update_history = [{}]
device_list = {
        1: {'name': 'espresso-machine', 'minpow': 800},
        2: {'name': 'washing-machine', 'minpow': 500},
        4: {'name': 'dish-washer', 'minpow': 500},
        8: {'name': 'induction-cooker', 'minpow': 800},
        16: {'name': 'irrigation-system', 'minpow': 400},
        32: {'name': 'oven', 'minpow': 800},
        64: {'name': 'microwave', 'minpow': 800},
        128: {'name': 'kitchen-light', 'minpow': 200},
        256: {'name': 'living-room-light', 'minpow': 200},
        512: {'name': 'dining-room-light', 'minpow': 200},
        1024: {'name': 'ground-floor-light', 'minpow': 200},
        2048: {'name': 'upper-floor-light', 'minpow': 200}}

# Enable CORS
_allow_origin = '*'
_allow_methods = 'PUT, GET, POST, DELETE, OPTIONS'
_allow_headers = 'Authorization, Origin, Accept, Content-Type, X-Requested-With'

@app.hook('after_request') #TippNicolas
def enable_cors():
    #Add headers to enable CORS
    bottle.response.headers['Access-Control-Allow-Origin'] = _allow_origin
    bottle.response.headers['Access-Control-Allow-Methods'] = _allow_methods
    bottle.response.headers['Access-Control-Allow-Headers'] = _allow_headers

@app.route('/', method = 'OPTIONS')
@app.route('/<path:path>', method = 'OPTIONS')
def options_handler(path = None):
    return

def log_measurement(device_id=0, command_str='', ts_min=0, ts_max=0, message_str=''):
    with open('logbook_measurements.csv', 'w', newline='') as csvfile:
        fieldnames = ['log time', 'device ID', 'device name', 'command', 'min timestamp', 'max timestamp', 'min datetime', 'max datetime', 'message']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            'log time' : datetime.datetime.now().astimezone().isoformat(), 
            'device id' : device_id,
            'device name' : device_list[device_id]['name'], 
            'command' : command_str,
            'min timestamp' : ts_min, 
            'max timestamp' : ts_max, 
            'min datetime' : datetime.fromtimestamp(ts_min), 
            'max datetime' : datetime.fromtimestamp(ts_max), 
            'message' : message_str]
            })


def connect_mysql() :
    # reload .env to exchange keras models on-the-fly
    try:
        env_file = "./.env"
        config.__init__(RepositoryEnv(env_file))
    except:
        logging.error(f"environment file {env_file} not found", exc_info=True)
    try:
        return pymysql.connect(
            host='localhost',
            user=config('myuser'),
            password=config('mypassword'),
            database=config('mydatabase'),
            cursorclass=pymysql.cursors.DictCursor)
    except:
        logging.error("could not open database at localhost", exc_info=True)    

@app.route('/show/<ts_from>/<ts_to>', method=['GET', 'OPTIONS'], name='show')
def show_device_ids(ts_from, ts_to):
    conn = connect_mysql()
    cur = conn.cursor()
    cur.execute('SELECT * FROM data WHERE timestamp >= "%s" AND timestamp <= "%s";',
               (float(ts_from), float(ts_to)))
    row = cur.fetchone()
    s = ''
    i = 0
    d = {}
    while row:
        i += 1
        s += '<p>{{t'+str(i)+'}}, {{v'+str(i)+'}}, {{d'+str(i)+'}}</p>'
        d['t' + str(i)] = row['timestamp']
        d['v' + str(i)] = row['value']
        d['d' + str(i)] = row['device']
        row = cur.fetchone()
    conn.close()
    return bottle.template(s, **d)


@app.route('/device_data/<device_id>/<data_start>', method=['GET'], name='device_data')
def get_device_data(device_id, data_start): # -> dict[str, str, str]:
    conn = connect_mysql()
    cur = conn.cursor()
    #device & id = id too slow. alternative column for each device_id, but major reformatting of database
    #cur.execute('SELECT timestamp, value, device FROM data WHERE timestamp >= "%s" AND device & "%s" = "%s";', (float(data_start), int(device_id), int(device_id)))
    cur.execute('SELECT timestamp, value, device FROM data WHERE timestamp >= "%s" AND device = "%s";', (float(data_start), int(device_id)))
    rows = cur.fetchall()
    conn.close()
    if bool(rows):
        logging.info("Data found and sent")
    else:
        logging.info("No data found")
    return json.dumps(rows)


@app.route('/diskspace', method=['GET'], name='diskspace') #TippNicolas -> POST
def get_remaining_disk_space(): # -> dict[str, str]: #TippNicolas "->"
    KB = 1024
    MB = 1024 * KB
    GB = 1024 * MB
    total, used, free = shutil.disk_usage('/')
    used_percent = used / total * 100
    response =  {"used_percent" : '{:,}'.format(round(used_percent)) + "%", "free" : '{:,}'.format(round(free / KB)) + " KB"}
    logging.info(response["used_percent"] + " and " +  response["free"])
    return response


@app.route('/classification/<ts_from_str>/<ts_to_str>/<window_length_str>', method=['GET'], name='classification')
def get_identified_devices(ts_from_str, ts_to_str, window_length_str): # -> dict[str, str]:
    conn = connect_mysql()
    cur = conn.cursor()
    cur.execute('SELECT * FROM data WHERE timestamp >= "%s" AND timestamp <= "%s";', (float(ts_from_str), float(ts_to_str)))
    data_list = cur.fetchall()
    conn.close()

    x = np.array([])
    xx = np.array([])
    for row in data_list:
            x = np.append(x, row['value'])

    window_length = int(config('keras_window_length'))

    i = 0 + window_length
    while i < x.size:
            xx = np.append(xx, x[i - window_length: i])
            i = i + window_length

    xx = xx.reshape((xx.size // window_length, window_length))
    model = keras.models.load_model(config('keras_filename'))
    yy = model.predict(xx)

    identified_devices = np.array([])
    for i in range(yy.shape[0]):
            identified_devices = np.append(identified_devices, np.argmax(yy[i]))
            identified_devices = np.unique(identified_devices)

    response = {}
    for i in identified_devices:
            id = 2**int(i)
            response[str(id)] = device_list[id]['name']

    logging.info(response)
    return response


@app.route('/update/<ts_from>/<ts_to>/<device_id>', method=['POST'], name='update')
def update_device_ids(ts_from, ts_to, device_id): # -> dict[str, str]:
    conn = connect_mysql()
    cur = conn.cursor()
    try:
        #count amount of all data points
        amount_selected = cur.execute('SELECT * FROM data WHERE timestamp >= "%s" AND timestamp <= "%s";', (float(ts_from), float(ts_to)))
        #update data points which don't include device_id yet
        amount_written = cur.execute('UPDATE data SET device = device | "%s" WHERE timestamp >= "%s" AND timestamp <= "%s";', (int(device_id), float(ts_from), float(ts_to)))
        #commit update to db
        conn.commit() #https://stackoverflow.com/questions/41916569/cant-write-into-mysql-database-from-python
        #log change for potential undo
        update_history.append({"device_id" : device_id, "ts_from" : ts_from, "ts_to" : ts_to})
        logging.info('UPDATE: size of update history now: ', len(update_history))
        #count amount of data points including device_id
        amount_committed = cur.execute('SELECT * FROM data WHERE timestamp >= "%s" AND timestamp <= "%s" AND device & "%s" = "%s";', (float(ts_from), float(ts_to), int(device_id), int(device_id)))
        conn.close()
    except pymysql.Error as e:
        logging.error('Got error {!r}, errno is {}'.format(e, e.args[0]), exc_info=True)

    logging.info(f'selected: {amount_selected}, written: {amount_written}, committed: {amount_committed}')
    #check if update was committed successfully
    response = {}
    if amount_selected == amount_committed:
        response['status'] = f'Device ID {device_id} ({device_list[int(device_id)]["name"]}) successfully written to database for all {amount_selected} data points by adding {amount_written} data points.'
        add2report(response['status'])
    else:
        response['status'] = f'Device ID {device_id} ({device_list[int(device_id)]["name"]}) could not be written to database ... only {amount_committed} of {amount_selected} data points include the device. Please contact your SYSTEMADMIN !!!'
    logging.info(response)
    return response
        

@app.route('/update_undo', method=['POST'], name='update_undo')
def update_undo(): # -> dict[str, str]:
    if len(update_history) > 1 :
        conn = connect_mysql()
        cur = conn.cursor()
        print('UPDATE_UNDO: clear device_id ' + update_history[-1]["device_id"] + ' from ' + update_history[-1]["ts_from"] + ' till ' + update_history[-1]["ts_to"])
        amount_selected = cur.execute('SELECT * FROM data WHERE timestamp >= "%s" AND timestamp <= "%s";', (float(update_history[-1]["ts_from"]), float(update_history[-1]["ts_to"])))
        amount_written  = cur.execute('UPDATE data SET device = device & ~"%s" WHERE timestamp >= "%s" AND timestamp <= "%s";',   (int(update_history[-1]["device_id"]), float(update_history[-1]["ts_from"]), float(update_history[-1]["ts_to"])))
        conn.commit()
        amount_committed = cur.execute('SELECT * FROM data WHERE timestamp >= "%s" AND timestamp <= "%s" AND device & "%s" = 0;', (float(update_history[-1]["ts_from"]), float(update_history[-1]["ts_to"]), int(update_history[-1]["device_id"])))
        conn.close()
        response = {}
        if amount_selected == amount_committed:
                response['status'] = f'selected == committed for Device ID {update_history[-1]["device_id"]} ({device_list[int(update_history[-1]["device_id"])]["name"]}) successfully removed for all {amount_selected} data points by changing {amount_written} data points.'
                update_history.pop() # remove last item
        else:
                response['status'] = f'selected != committed for Device ID {update_history[-1]["device_id"]} ({device_list[int(update_history[-1]["device_id"])]["name"]}) could not be written to database ... still {amount_selected - amount_committed} data points include the device. Please contact your SYSTEMADMIN !!!'
        logging.info(response)
        return response
    else:
        response = {}
        response['status'] = f'UPDATE_UNDO: not possible since no update history available'
        logging.warning(response)
        return response


if __name__ == "__main__":
    try:
        # 'gevent' opens many threads to handle async. alternative: 'gunicorn'
        logging.info(f"Bottle starting GeventServer at {config('myhost') + ':' + config('myport')}")
        app.run(server='gevent', host=config('myhost'), port=config('myport'), debug=True)
    except:
        logging.error(f"Bottle failed to start GeventServer at {config('myhost') + ':' + config('myport')}", exc_info=True)