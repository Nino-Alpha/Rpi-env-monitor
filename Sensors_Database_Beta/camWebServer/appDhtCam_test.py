# 2025-03-10 ningchy
# appDhtCam + Freq ：Done.
# +alarm system :testing
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
from flask import Flask, render_template, send_file, make_response, request, g, Response,redirect
import sqlite3
from camera_pi2 import Camera
import time
from datetime import datetime 

app = Flask(__name__)

# 获取数据库连接 
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect('../sensorsData.db')
    return g.db
# 关闭数据库连接
@app.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None) # 从 g 对象中移除 db 键，如果没有则返回 None。
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

def getHistData(numSamples):
    db = get_db()
    curs = db.cursor()
    curs.execute("SELECT * FROM DHT_data ORDER BY timestamp DESC LIMIT ?", (numSamples,))
    data = curs.fetchall()
    dates, temps, hums = [], [], []
    for row in reversed(data):
        dates.append(row[0])
        temps.append(row[1])
        hums.append(row[2])
    return dates, temps, hums

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

# 在WebServer.py中添加数据库初始化（只需执行一次）
# def init_db():
#     db = sqlite3.connect('../sensorsData.db')
#     curs = db.cursor()
#     curs.execute('''CREATE TABLE IF NOT EXISTS thresholds
#         (period TEXT PRIMARY KEY,
#         start_time TEXT NOT NULL,
#         end_time TEXT NOT NULL,
#         temp_threshold REAL NOT NULL,
#         hum_threshold REAL NOT NULL)''')
#     # 插入默认值
#     curs.execute('''INSERT OR IGNORE INTO thresholds VALUES 
#         ('A', '00:00', '10:00', 30.0, 40.0),
#         ('B', '10:01', '15:00', 30.0, 40.0),
#         ('C', '15:01', '23:59', 30.0, 40.0)''')
#     db.commit()
#     return None

# 根据当前系统时间获取当前时段阈值
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
    
# getThresholds() 函数用于获取所有时段的阈值
def getThresholds():
    db = get_db()
    curs = db.cursor()
    curs.execute("SELECT * FROM thresholds ORDER BY period")
    return curs.fetchall()

# 主页路由
@app.route("/")
def index():
    time, temp, hum = getLastData()

    sampleFreq = getLastFreq()  # 获取最新频率
    
    current_th = get_current_threshold() # 获取当前时段阈值

    thresholds = getThresholds() # 获取所有时段阈值

    templateData = {
        'time': time,
        'temp': temp,
        'hum': hum,
        'numSamples': 100,  
        'sampleFreq' : sampleFreq, #检测频率
        'current_th': current_th,
        'thresholds': thresholds
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

# 参数提交路由1：历史数据条数
@app.route('/set_numSamples', methods=['POST']) ###
def my_form_post():
    global numSamples # 全局变量
    numSamples = int(request.form['numSamples'])
    numMaxSamples = maxRowsTable()
    if numSamples > numMaxSamples:
        numSamples = numMaxSamples - 1

    time, temp, hum = getLastData() 
    templateData = {
        'time': time,
        'temp': temp,
        'hum': hum,
        'numSamples': numSamples
    }
    return render_template('index.html', **templateData)

# 参数提交路由2 ：数据检测频率
@app.route('/set_frequency', methods=['POST'])
def set_frequency():
    global sampleFreq # 全局变量
    new_freq = int(request.form['frequency']) # 获取用户输入的频率---<html>中的name 
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
    numSamples = int(request.args.get('numSamples', 100))  
    times, temps, hums = getHistData(numSamples)
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title("Temperature [°C]")
    axis.set_xlabel("Samples")
    axis.grid(True)
    xs = range(numSamples)
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
    numSamples = int(request.args.get('numSamples', 100))  
    times, temps, hums = getHistData(numSamples)
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title("Humidity [%]")
    axis.set_xlabel("Samples")
    axis.grid(True)
    xs = range(numSamples)
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