import socket
from decouple import Config, RepositoryEnv, Csv
from gevent import monkey; monkey.patch_all() # https://bottlepy.org/docs/dev/async.html
import bottle
import pymysql
from bottle import response, post, hook
#from bottle_cors_plugin import cors_plugin
import json
import shutil
import numpy as np # https://www.nbshare.io/notebook/505221353/ERROR-Could-not-find-a-version-that-satisfies-the-requirement-numpy==1-22-3/
import tensorflow as tf # https://qengineering.eu/install-tensorflow-2.2.0-on-raspberry-pi-4.html
import tensorflow.keras as keras

config = Config(RepositoryEnv("./.env"))

app = bottle.Bottle()
#app.install(cors_plugin('*'))

update_history = [{}]

def connect_mysql() :
    # reload .env to exchange keras models on-the-fly
    config.__init__(RepositoryEnv("./.env"))
    return pymysql.connect(
        host='localhost',
        user=config('myuser'),
        password=config('mypassword'),
        database=config('mydatabase'),
        cursorclass=pymysql.cursors.DictCursor)

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


@app.route('/show/<ts_from>/<ts_to>', method=['GET', 'OPTIONS'], name='show')
def show_device_ids(ts_from, ts_to):
    conn = connect_mysql()
    cur = conn.cursor()
    cur.execute('SELECT * FROM data WHERE timestamp > "%s" AND timestamp < "%s";',
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


@app.route('/device_data/<device_id>/<data_start>', method=['GET', 'OPTIONS'], name='device_data')
def get_device_data(device_id, data_start): # -> dict[str, str, str]:
    conn = connect_mysql()
    cur = conn.cursor()
    cur.execute('SELECT timestamp, value, device FROM data WHERE timestamp > "%s" AND device & "%s" = 1;', (float(data_start), int(device_id)))
    rows = cur.fetchall()
    conn.close()
    return json.dumps(rows)


@app.route('/diskspace', method=['GET', 'OPTIONS'], name='diskspace') #TippNicolas -> POST
def get_remaining_disk_space(): # -> dict[str, str]: #TippNicolas "->"
    KB = 1024
    MB = 1024 * KB
    GB = 1024 * MB
    total, used, free = shutil.disk_usage('/')
    used_percent = used / total * 100
    response =  {"used_percent" : '{:,}'.format(round(used_percent)) + "%", "free" : '{:,}'.format(round(free / KB)) + " KB"}
    print(response["used_percent"] + " and " +  response["free"])
    return response


@app.route('/classification/<ts_from_str>/<ts_to_str>/<window_length_str>', method=['GET', 'OPTIONS'], name='classification')
def get_identified_devices(ts_from_str, ts_to_str, window_length_str): # -> dict[str, str]:
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
        2048: {'name': 'upper-floor-light', 'minpow': 200},
    }

    conn = connect_mysql()
    cur = conn.cursor()
    cur.execute('SELECT * FROM data WHERE timestamp > "%s" AND timestamp < "%s";',
               (float(ts_from_str), float(ts_to_str)))
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

    print(response)
    return response


@app.route('/update/<ts_from>/<ts_to>/<device_id>', method=['POST', 'OPTIONS'], name='update')
def update_device_ids(ts_from, ts_to, device_id): # -> dict[str, str]:
    try:
        conn = connect_mysql()
        cur = conn.cursor()
        amount_written = cur.execute('UPDATE data SET device = device | "%s" WHERE timestamp > "%s" AND timestamp < "%s";', (int(device_id), float(ts_from), float(ts_to)))
        print('written data: ', amount_written)
        conn.commit() #https://stackoverflow.com/questions/41916569/cant-write-into-mysql-database-from-python
        update_history.append({"device_id" : device_id, "ts_from" : ts_from, "ts_to" : ts_to})
        print('UPDATE: size of update history now: ', len(update_history))
        amount_count = cur.execute('SELECT COUNT(device) FROM data WHERE timestamp > "%s" AND timestamp < "%s" AND device & "%s" = 1;', (float(ts_from), float(ts_to), int(device_id)))        
        print('read data: ', amount_count)
        conn.close()
        if amount_written == amount_count:
            return bottle.HTTPResponse(body = 'Device ID ' + device_id + ' successfully written to database for ' + str(amount_written) + ' seconds.', status = 200)
        else:
            return bottle.HTTPResponse(body = 'Device ID ' + device_id + ' could not be written to database ... please contact your SYSTEMADMIN !!!', status = 500)
    except MySQLError as e:
        print('Got error {!r}, errno is {}'.format(e, e.args[0]))
    

@app.route('/update_undo', method=['POST', 'OPTIONS'], name='update_undo')
def update_undo(): # -> dict[str, str]:
    if len(update_history) > 1 :
        conn = connect_mysql()
        cur = conn.cursor()
        print('UPDATE_UNDO: clear device_id ' + update_history[-1]["device_id"] + ' from ' + update_history[-1]["ts_from"] + ' till ' + update_history[-1]["ts_to"])
        amount_written = cur.execute('UPDATE data SET device = device & ~"%s" WHERE timestamp > "%s" AND timestamp < "%s";', (int(update_history[-1]["device_id"]), float(update_history[-1]["ts_from"]), float(update_history[-1]["ts_to"])))
        conn.commit()
        update_history.pop() # remove last item
        amount_count = cur.execute('SELECT COUNT(device) FROM data WHERE timestamp > "%s" AND timestamp < "%s" AND device & "%s" = 0;', (float(ts_from), float(ts_to), int(device_id)))        
        conn.close()
        if amount_written == amount_count:
            return bottle.HTTPResponse(body = 'Device ID ' + device_id + ' successfully deleted from database for ' + str(amount_written) + ' seconds.', status = 200)
        else:
            return bottle.HTTPResponse(body = 'Device ID ' + device_id + ' could not be deleted from database ... please contact your SYSTEMADMIN !!!', status = 500)
    else :
        print('UPDATE_UNDO: not possible since no update history available')
        return bottle.HTTPResponse(body = 'UPDATE_UNDO: not possible since no update history available', status = 500)


if __name__ == "__main__":
    # 'gevent' opens many threads to handle async. alternative: 'gunicorn'
    app.run(server='gevent', host=config('myhost'), port=config('myport'), debug=True)
