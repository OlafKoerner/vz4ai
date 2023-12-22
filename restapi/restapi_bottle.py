from gevent import monkey; monkey.patch_all() # https://bottlepy.org/docs/dev/async.html
import bottle
import bottle_mysql
from bottle import response, request, post
import shutil

app = bottle.Bottle()
# dbhost is optional, default is localhost
plugin = bottle_mysql.Plugin(dbuser='vzlogger', dbpass='demo', dbname='volkszaehler')
app.install(plugin)

update_history = [{}]

@app.route('/show/<ts_from>/<ts_to>')
def show_device_ids(ts_from, ts_to, db):
    db.execute('SELECT * FROM data WHERE timestamp > "%s" AND timestamp < "%s" LIMIT 1000;',
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
    return bottle.template(s, **d)
    #return bottle.HTTPError(404, "Page not found")

@app.route('/update/<ts_from>/<ts_to>/<device_id>')
def update_device_ids(ts_from, ts_to, device_id, db):
    db.execute('UPDATE data SET device = device | "%s" WHERE timestamp > "%s" AND timestamp < "%s" LIMIT 100000;', (int(device_id), float(ts_from), float(ts_to)))
    update_history.append({"device_id" : device_id, "ts_from" : ts_from, "ts_to" : ts_to})
    print('UPDATE: size of update history now: ', len(update_history))
    return bottle.HTTPResponse(status = 200)

@app.route('/update_undo')
def update_undo(db):
	if len(update_history) > 1 :
		print('UPDATE_UNDO: clear device_id ' + update_history[-1]["device_id"] + ' from ' + update_history[-1]["ts_from"] + ' till ' + update_history[-1]["ts_to"])
		db.execute('UPDATE data SET device = device & ~"%s" WHERE timestamp > "%s" AND timestamp < "%s" LIMIT 100000;', (int(update_history[-1]["device_id"]), float(update_history[-1]["ts_from"]), float(update_history[-1]["ts_to"])))
		update_history.pop() # remove last item
		return bottle.HTTPResponse(status = 200)
	else : 
		print('UPDATE_UNDO: not possible since no update history available') 
		return bottle.HTTPResponse(body = 'UPDATE_UNDO: not possible since no update history available', status = 500)

@app.route('/diskspace')
def get_remaining_disk_space():
	KB = 1024
	MB = 1024 * KB
	GB = 1024 * MB
	total, used, free = shutil.disk_usage('/')
	used_percent = used / total * 100
	response =  {"used_percent" : '{:,}'.format(round(used_percent)) + "%", "free" : '{:,}'.format(round(free / KB)) + " KB"}
	print(response["used_percent"] + " and " +  response["free"])
	return response

def main():
    app.run(host='192.168.178.185', port=8082, debug=True, server='gevent')
    while 1: 1

if __name__ == "__main__":
    main()
