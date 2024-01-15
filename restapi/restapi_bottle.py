from decouple import Config, RepositoryEnv, Csv
from gevent import monkey; monkey.patch_all() # https://bottlepy.org/docs/dev/async.html
import bottle
import pymysql
from bottle import response, post, hook
#from bottle_cors_plugin import cors_plugin
import shutil
import numpy as np
import tensorflow as tf
import keras

app = bottle.Bottle()
#app.install(cors_plugin('*'))

update_history = [{}]

def connect_mysql() :
    config = Config(RepositoryEnv("/Users/Olaf/Documents/GitHub/vz4ai/restapi/.env"))
    return pymysql.connect(
    host=config('myhost'),
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

@app.route('/show/<ts_from>/<ts_to>', method=['POST', 'OPTIONS'])
def show_device_ids(ts_from, ts_to):
    conn = connect_mysql()
    cur = conn.cursor()
    cur.execute('SELECT * FROM data WHERE timestamp > "%s" AND timestamp < "%s" LIMIT 1000;',
               (float(ts_from), float(ts_to)))
    row = db.fetchone()
    s = ''
    i = 0
    d = {}
    while row:
        i += 1
        s += '<p>{{t'+str(i)+'}}, {{v'+str(i)+'}}, {{d'+str(i)+'}}</p>'
        d['t' + str(i)] = row['timestamp']
        d['v' + str(i)] = row['value']
        d['d' + str(i)] = row['device']
        row = db.fetchone()
    conn.close()
    return bottle.template(s, **d)
    #return bottle.HTTPError(404, "Page not found")

@app.route('/update/<ts_from>/<ts_to>/<device_id>', method=['POST', 'OPTIONS'])
def update_device_ids(ts_from, ts_to, device_id):
    conn = connect_mysql()
    cur = conn.cursor()
    cur.execute('UPDATE data SET device = device | "%s" WHERE timestamp > "%s" AND timestamp < "%s" LIMIT 100000;', (int(device_id), float(ts_from), float(ts_to)))
    update_history.append({"device_id" : device_id, "ts_from" : ts_from, "ts_to" : ts_to})
    print('UPDATE: size of update history now: ', len(update_history))
    conn.clode()
    return bottle.HTTPResponse(status = 200)

@app.route('/update_undo', method=['GET', 'POST', 'OPTIONS'])
def update_undo():
    if len(update_history) > 1 :
        conn = connect_mysql()
        cur = conn.cursor()
        print('UPDATE_UNDO: clear device_id ' + update_history[-1]["device_id"] + ' from ' + update_history[-1]["ts_from"] + ' till ' + update_history[-1]["ts_to"])
        cur.execute('UPDATE data SET device = device & ~"%s" WHERE timestamp > "%s" AND timestamp < "%s" LIMIT 100000;', (int(update_history[-1]["device_id"]), float(update_history[-1]["ts_from"]), float(update_history[-1]["ts_to"])))
        update_history.pop() # remove last item
        conn.close()
        return bottle.HTTPResponse(status = 200)
    else :
        print('UPDATE_UNDO: not possible since no update history available')
        return bottle.HTTPResponse(body = 'UPDATE_UNDO: not possible since no update history available', status = 500)

@app.route('/diskspace', method=['GET', 'POST', 'OPTIONS']) #TippNicolas -> POST
def get_remaining_disk_space() -> dict[str, str]: #TippNicolas "->"
	KB = 1024
	MB = 1024 * KB
	GB = 1024 * MB
	total, used, free = shutil.disk_usage('/')
	used_percent = used / total * 100
	response =  {"used_percent" : '{:,}'.format(round(used_percent)) + "%", "free" : '{:,}'.format(round(free / KB)) + " KB"}
	print(response["used_percent"] + " and " +  response["free"])
	return response

@app.route('/classification/<ts_from_str>/<ts_to_str>/<window_length_str>', method=['GET', 'POST', 'OPTIONS'])
def get_identified_devices(ts_from_str, ts_to_str, window_length_str) -> dict[str, str]:
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
    cur.execute('SELECT * FROM data WHERE timestamp > "%s" AND timestamp < "%s" LIMIT 10000;',
               (float(ts_from_str), float(ts_to_str)))
    data_list = cur.fetchall()
    conn.close()

    x = np.array([])
    xx = np.array([])
    for row in data_list:
        x = np.append(x, row['value'])

    window_length = int(window_length_str)
    i = 0 + window_length
    while i < x.size:
        xx = np.append(xx, x[i - window_length: i])
        i = i + window_length

    xx = xx.reshape((xx.size // window_length, window_length))

    model = keras.models.load_model('Devices4Data10000Epoch1000.keras')
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

if __name__ == "__main__":
    # 'gevent' opens many threads to handle async. alternative: 'gunicorn'
    app.run(server='gevent', host='127.0.0.1', port=8082, debug=True)
    #app.run(server='gevent', host='192.168.178.185', port=8082, debug=True)
