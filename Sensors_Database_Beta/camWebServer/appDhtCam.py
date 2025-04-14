# 2025-03-10 ningchy
# appDhtCam + Freq ：Done.
# +alarm system :testing
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
from flask import Flask
from flask import render_template
from flask import make_response
from flask import request
from flask import g
from flask import Response
from flask import redirect
from flask import session
import sqlite3
from camera_pi2 import Camera
import time
from datetime import datetime 
#from Sensors_Database_Beta import logDHT

app = Flask(__name__)
app.secret_key = '8888'  # 设置session密钥
# 获取数据库连接 
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect('../sensorsData.db')
    return g.db
# 关闭数据库连接
@app.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None) 
    if db is not None:
        db.close()

def getLastData():
    db = get_db()
    curs = db.cursor()
    curs.execute("SELECT * FROM DHT_data ORDER BY timestamp DESC LIMIT 1")
    row = curs.fetchone()
    if row:
        time, temp, hum = str(row[0]), row[1], row[2]
        return time, temp, hum
    return None, None, None

# def getHistData(numSamples):
#     db = get_db()
#     curs = db.cursor()
#     curs.execute("SELECT * FROM DHT_data ORDER BY timestamp DESC LIMIT ?", (numSamples,))
#     data = curs.fetchall()
#     dates, temps, hums = [], [], []
#     for row in reversed(data):
#         dates.append(row[0])
#         temps.append(row[1])
#         hums.append(row[2])
#     return dates, temps, hums

def maxRowsTable():
    db = get_db()
    curs = db.cursor()
    curs.execute("SELECT COUNT(temp) FROM DHT_data")
    maxNumberRows = curs.fetchone()[0] 
    return maxNumberRows

def gen(camera): 
    
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
# 获取最新频率        
def getLastFreq():
    db = get_db()
    curs = db.cursor()
    curs.execute("SELECT * FROM Sample_freq ORDER BY timestamp DESC LIMIT 1")
    row = curs.fetchone()
    if row:
        freq = row[1]
        return freq
    return None
# 频率写入数据库  
def logFreq(sampleFreq):
    db = get_db()
    curs = db.cursor()
    curs.execute("INSERT INTO Sample_freq values(datetime('now','localtime'), (?))", (sampleFreq,))
    db.commit()
    return None


def get_current_threshold():
    current_time = datetime.now().strftime("%H:%M")
    db = get_db()
    curs = db.cursor()
    curs.execute('''SELECT * FROM thresholds 
        WHERE start_time <= ? AND end_time >= ? 
        ORDER BY period LIMIT 1''', (current_time, current_time))
    return curs.fetchone()

# 更新时段阈值
def update_thresholds(period, start, end, temp, hum):
    db = get_db()
    curs = db.cursor()
    curs.execute('''UPDATE thresholds SET 
        start_time=?, end_time=?, temp_threshold=?, hum_threshold=?
        WHERE period=?''', (start, end, temp, hum, period))
    db.commit()


# 阈值设置路由
@app.route('/set_thresholds', methods=['POST'])
def set_thresholds():
    try:
        # 验证并更新三个时段
        for period in ['A', 'B', 'C']:
            start = request.form[f'{period}_start']
            end = request.form[f'{period}_end']
            temp = float(request.form[f'{period}_temp'])
            hum = float(request.form[f'{period}_hum'])
            
            # 阈值范围验证
            if not (0.0 <= temp <= 50.0) or not (10.0 <= hum <= 80.0):
                raise ValueError("阈值超出合法范围")
                
            update_thresholds(period, start, end, temp, hum)
            
        return redirect('/')
    except Exception as e:
        return f"<script>alert('设置错误: {str(e)}');window.history.back();</script>"
    

def getThresholds():
    db = get_db()
    curs = db.cursor()
    curs.execute("SELECT * FROM thresholds ORDER BY period")
    return curs.fetchall()

# 主页路由
@app.route("/")
def index():
    time, temp, hum = getLastData()
    sampleFreq = getLastFreq()  
    current_th = get_current_threshold() 
    thresholds = getThresholds() 
    selected_date = session.get('selected_date', None)

    templateData = {
        'time': time,
        'temp': temp,
        'hum': hum, 
        'sampleFreq' : sampleFreq, 
        'current_th': current_th,
        'thresholds': thresholds,
        'selected_date': selected_date
    }
    return render_template('index.html', **templateData)

# 视频页跳转路由
@app.route('/camera') 
def cam():
	"""Video streaming home page."""
	timeNow = time.asctime( time.localtime(time.time()) )
	templateData = {
      'time': timeNow
	}
	return render_template('camera.html', **templateData)

@app.route('/video_feed') 
def video_feed():
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 数据库查询路由
@app.route('/query_history', methods=['POST'])
def query_history():
    selected_date = request.form['date']
    start_time = request.form['start_time']
    end_time = request.form['end_time']
    
    session['selected_date'] = selected_date
    query = """
        SELECT timestamp, temp, hum
        FROM DHT_data
        WHERE DATE(timestamp) = ?
        AND TIME(timestamp) BETWEEN ? AND ?
    """
    db = get_db()
    curs = db.cursor()
    curs.execute(query, (selected_date, start_time, end_time))
    rows = curs.fetchall()

    templateData = {
        'data': rows,
        'start_time': start_time,
        'end_time': end_time,
        'selected_date': selected_date
    }
    return render_template('table.html', **templateData)

#数据库统计路由
@app.route('/select_graph', methods=['POST'])
def select_graph():
    selected_date = request.form['date']
    start_time = request.form['start_time']
    end_time = request.form['end_time']
    
    query = """
        SELECT timestamp, temp, hum
        FROM DHT_data
        WHERE DATE(timestamp) = ?
        AND TIME(timestamp) BETWEEN ? AND ?
    """
    db = get_db()
    curs = db.cursor()
    curs.execute(query, (selected_date, start_time, end_time))
    rows = curs.fetchall()
    
    # 处理数据以生成图表
    times, temps, hums = [], [], []
    for row in rows:
        times.append(row[0])
        temps.append(row[1])
        hums.append(row[2])
    templateData = {
        'times': times,
        'temps': temps,
        'hums': hums, 
        'start_time': start_time,
        'end_time': end_time,
        'selected_date': selected_date
    }
    return render_template('graphs.html', **templateData)
#中转路由
@app.route('/table_graphs', methods=['POST'])
def table_graphs(): 
    #从table.html接收数据
    rows = request.form.getlist('rows[]') 
    selected_date = request.form['selected_date']
    start_time = request.form['start_time']
    end_time = request.form['end_time']
    # 处理数据以生成图表
    times, temps, hums = [], [], []
    for row in rows:
        times.append(row[0])
        temps.append(row[1])
        hums.append(row[2])

    templateData = {
        'times': times,
        'temps': temps,
        'hums': hums, 
        'start_time': start_time,
        'end_time': end_time,
        'selected_date': selected_date
    }
    return render_template('graphs.html', **templateData)
# 参数提交路由2 ：数据检测频率
@app.route('/set_frequency', methods=['POST'])
def set_frequency():
    global sampleFreq 
    new_freq = int(request.form['frequency']) 
    if new_freq > 0:
        sampleFreq = new_freq
    # 将新的频率写入数据库
    logFreq(sampleFreq)
    # 返回最新频率
    sampleFreq = getLastFreq()
    # 返回最新数据
    time, temp, hum = getLastData()
    templateData = {
        'time': time,
        'temp': temp,
        'hum': hum,
        'sampleFreq': sampleFreq   ###
    }
    return render_template('index.html', **templateData)

# 温度图表路由
@app.route('/plot/temp')
def plot_temp():
    # numSamples = int(request.args.get('numSamples', 100))  
    # times, temps, hums = getHistData(numSamples) 由于不再依赖输入样本数。所以要通过网页直接获取数据。
    # 从html接收times, temps, hums
    temps = request.args.getlist('temps[]')
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title("Temperature [°C]")
    axis.set_xlabel("Samples")
    axis.grid(True)
    xs = range(len(temps))   # 要将x轴的范围改为len(temps)。
    axis.plot(xs, temps)
    canvas = FigureCanvas(fig)
    output = io.BytesIO()
    canvas.print_png(output)
    response = make_response(output.getvalue())
    response.mimetype = 'image/png'
    return response
# 湿度图表路由
@app.route('/plot/hum')
def plot_hum():
    # numSamples = int(request.args.get('numSamples', 100))  
    # times, temps, hums = getHistData(numSamples)

    hums = request.args.getlist('hums[]')
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title("Humidity [%]")
    axis.set_xlabel("Samples")
    axis.grid(True)
    xs = range(len(hums))
    axis.plot(xs, hums)
    canvas = FigureCanvas(fig)
    output = io.BytesIO()
    canvas.print_png(output)
    response = make_response(output.getvalue())
    response.mimetype = 'image/png'
    return response

#获取第一条数据（作为系统启动时间）
def getFirstData():
    db = get_db()
    curs = db.cursor()
    curs.execute("SELECT * FROM DHT_data ORDER BY timestamp ASC LIMIT 1")
    row = curs.fetchone()
    if row:
        time, temp, hum = str(row[0]), row[1], row[2]
        return time, temp, hum
    return None, None, None

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True, threaded=True)