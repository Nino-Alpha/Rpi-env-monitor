# 2025-03-10 ningchy
# appDhtCam + Freq ：Done.
# +alarm system :testing
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
from flask import Flask
from flask import render_template
from flask import request
from flask import g
from flask import Response
from flask import redirect
from flask import session
from flask import jsonify
from flask import json
import sqlite3
from camera_pi2 import Camera
import time
from datetime import datetime 
import base64
import plotly.graph_objs as go
import plotly.offline as pyo
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from threading import Thread
import numpy as np
from sklearn.preprocessing import MinMaxScaler  
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense #需要虚拟环境兼容运行。py3.10/py3.9
from pmdarima import auto_arima 

app = Flask(__name__)
# 设置session密钥
app.secret_key = '8888'  
# 邮件配置
EMAIL_CONFIG = {
    'smtp_server': 'smtp.qq.com',  # SMTP服务器地址
    'smtp_port': 465,  # SMTP端口
    'sender_email': '1319716674@qq.com',  # 发件邮箱
    'sender_password': 'qbbotthcvjrdgfbc',  # 发件邮箱密码
    # 'use_ssl': True ,
    'receiver_email': 'chenyv287212@gmail.com',  # 收件邮箱
    'min_interval': 300  # 最小发送间隔，单位秒（5分钟）  
}

# 全局变量记录上次发送时间
last_email_sent = {
    'temperature': 0,
    'humidity': 0
}
# 全局变量记录上次报警时间
last_alarm_recorded = {
    'temperature': 0,
    'humidity': 0
}
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
# 获取最新数据
def getLastData():
    db = get_db()
    curs = db.cursor()
    curs.execute("SELECT * FROM DHT_data ORDER BY timestamp DESC LIMIT 1")
    row = curs.fetchone()
    if row:
        time, temp, hum = str(row[0]), row[1], row[2]
        return time, temp, hum
    return None, None, None
# 获取数据表行数
def maxRowsTable():
    db = get_db()
    curs = db.cursor()
    curs.execute("SELECT COUNT(temp) FROM DHT_data")
    maxNumberRows = curs.fetchone()[0] 
    return maxNumberRows
# 获取摄像头帧
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
# 获取所有时段阈值
def getThresholds():
    db = get_db()
    curs = db.cursor()
    curs.execute("SELECT * FROM thresholds ORDER BY period")
    return curs.fetchall()
# 获取当前时段阈值
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

# 邮件发送函数
def send_email(subject, content):
    try:
        # 检查是否在冷却时间内
        current_time = time.time()
        if subject == '温度报警' and current_time - last_email_sent['temperature'] < EMAIL_CONFIG['min_interval']:
            return
        if subject == '湿度报警' and current_time - last_email_sent['humidity'] < EMAIL_CONFIG['min_interval']:
            return
        msg = MIMEText(content, 'plain', 'utf-8')
        msg['Subject'] = Header(subject, 'utf-8')
        msg['From'] = EMAIL_CONFIG['sender_email']
        msg['To'] = EMAIL_CONFIG['receiver_email']

        with smtplib.SMTP_SSL(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']) as server:
            server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
            server.sendmail(EMAIL_CONFIG['sender_email'], [EMAIL_CONFIG['receiver_email']], msg.as_string())
            server.quit()  # 添加这行代码来正确关闭连接
    # 更新上次发送时间
        if subject == '温度报警':
            last_email_sent['temperature'] = current_time
        elif subject == '湿度报警':
            last_email_sent['humidity'] = current_time
    except Exception as e:
        print(f"邮件发送失败: {e}")

# 后台监控任务
def background_monitor():
    with app.app_context():  # 添加应用上下文
        while True:
            try:
                # 获取最新数据
                _, temp, hum = getLastData()
                current_th = get_current_threshold()

                if temp is not None and hum is not None and current_th:
                    # 检查温度
                    current_time = time.time()
                    if temp > current_th[3]:
                        if current_time - last_alarm_recorded['temperature'] >= 60:  # 确保每分钟只记录一次
                            send_email('温度报警', f'当前温度 {temp}℃ 超过阈值 {current_th[3]}℃')
                            log_alarm('temperature', temp, current_th[3])
                            last_alarm_recorded['temperature'] = current_time
                    # 检查湿度
                    if hum > current_th[4]:
                        if current_time - last_alarm_recorded['humidity'] >= 60:  # 确保每分钟只记录一次
                            send_email('湿度报警', f'当前湿度 {hum}% 超过阈值 {current_th[4]}%')
                            log_alarm('humidity', hum, current_th[4])
                            last_alarm_recorded['humidity'] = current_time
                # 每180秒检查一次
                time.sleep(180)
            except Exception as e:
                print(f"后台监控出错: {e}")
                time.sleep(60)

# 新增报警记录函数
def log_alarm(alarm_type, current_value, threshold):
    db = get_db()
    curs = db.cursor()
    curs.execute("INSERT INTO alarm_logs (timestamp, alarm_type, current_value, threshold) VALUES (datetime('now','localtime'), ?, ?, ?)",
                 (alarm_type, current_value, threshold))
    db.commit()

# 数据统计函数
def calculate_statistics(selected_date, start_time, end_time):
    db = get_db()
    curs = db.cursor()
    
    # 查询数据
    query = """
        SELECT temp, hum FROM DHT_data
        WHERE DATE(timestamp) = ? AND TIME(timestamp) BETWEEN ? AND ?
    """
    curs.execute(query, (selected_date, start_time, end_time))
    data = np.array(curs.fetchall())
    
    if len(data) == 0:
        return None
    
    temps = data[:, 0]
    hums = data[:, 1]
    
    # 处理离群值
    temps = handle_outliers(temps)
    hums = handle_outliers(hums)
    # 基础统计
    stats = {
        'temp_avg': float(np.mean(temps)),  # 转换为float
        'temp_min': float(np.min(temps)),
        'temp_max': float(np.max(temps)),
        'temp_median': float(np.median(temps)),
        'hum_avg': float(np.mean(hums)),
        'hum_min': float(np.min(hums)),
        'hum_max': float(np.max(hums)),
        'hum_median': float(np.median(hums)),
        'correlation': float(np.corrcoef(temps, hums)[0, 1]),
    }
    # 温度区间百分比
    temp_bins = [0, 10, 20, 30, 40, 50]
    temp_counts, _ = np.histogram(temps, bins=temp_bins)
    stats['temp_bins'] = {
        f'{temp_bins[i]}-{temp_bins[i+1]}': float(count/len(temps)*100)  # 转换为float
        for i, count in enumerate(temp_counts)
    }
    
    # 湿度区间百分比
    hum_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    hum_counts, _ = np.histogram(hums, bins=hum_bins)
    stats['hum_bins'] = {
        f'{hum_bins[i]}-{hum_bins[i+1]}': float(count/len(hums)*100)  # 转换为float
        for i, count in enumerate(hum_counts)
    }
    hourly_stats = calculate_hourly_statistics(selected_date, start_time, end_time) # 计算每小时统计数据
    for stat in hourly_stats:
        for key in stat:
            if stat[key] is not None and isinstance(stat[key], (np.integer, np.floating)):
                stat[key] = float(stat[key])
    # 生成温度和湿度的饼图数据结构
    temp_pie = {
        'data': [{
            'labels': list(stats['temp_bins'].keys()),
            'values': list(stats['temp_bins'].values()),
            'type': 'pie'
        }],
        'layout': {
            'title': '温度分布百分比'
        }
    }
    hum_pie = {
        'data': [{
            'labels': list(stats['hum_bins'].keys()),
            'values': list(stats['hum_bins'].values()),
            'type': 'pie'
        }],
        'layout': {
            'title': '湿度分布百分比'
        }
    }
    stats.update({
        'temp_pie': temp_pie,
        'hum_pie': hum_pie,          
        'hourly_stats': hourly_stats
        })

    return stats

# 统计结果保存到数据库
def save_stats_result(start_date, end_date, start_time, end_time, stats):
    db = get_db()
    curs = db.cursor()

    temp_bins = json.dumps(stats['temp_bins'])
    hum_bins = json.dumps(stats['hum_bins'])

    curs.execute('''INSERT INTO stats_results (
        start_date, end_date, start_time, end_time,
        temp_avg, temp_min, temp_max, temp_median,
        hum_avg, hum_min, hum_max, hum_median,
        correlation, temp_bins, hum_bins
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
        start_date, end_date, start_time, end_time,
        stats['temp_avg'], stats['temp_min'], stats['temp_max'], stats['temp_median'],
        stats['hum_avg'], stats['hum_min'], stats['hum_max'], stats['hum_median'],
        stats['correlation'], temp_bins, hum_bins
    ))
    db.commit()

# 计算每小时的统计数据
def calculate_hourly_statistics(selected_date, start_time, end_time):
    db = get_db()
    curs = db.cursor()
    
    # 解析开始和结束时间
    start_hour = int(start_time.split(':')[0])
    end_hour = int(end_time.split(':')[0])
    
    hourly_stats = []
    
    for hour in range(start_hour, end_hour):
        # 构建时间范围
        hour_start = f"{hour:02d}:00"
        hour_end = f"{hour+1:02d}:00"
        
        # 查询该小时的数据
        query = """
            SELECT temp, hum FROM DHT_data
            WHERE DATE(timestamp) = ? 
            AND TIME(timestamp) >= ? AND TIME(timestamp) < ?
        """
        curs.execute(query, (selected_date, hour_start, hour_end))
        data = np.array(curs.fetchall())
        
        if len(data) == 0:
            hourly_stats.append({
                'time_range': f"{hour_start}-{hour_end}",
                'temp_avg': None,
                'temp_min': None,
                'temp_max': None,
                'temp_median': None,
                'hum_avg': None,
                'hum_min': None,
                'hum_max': None,
                'hum_median': None
            })
            continue
            
        else:
            temps = data[:, 0]
            hums = data[:, 1]
            
            hourly_stats.append({
                'time_range': f"{hour_start}-{hour_end}",
                'temp_avg': float(np.mean(temps)) if len(temps) > 0 else None,
                'temp_min': float(np.min(temps)) if len(temps) > 0 else None,
                'temp_max': float(np.max(temps)) if len(temps) > 0 else None,
                'temp_median': float(np.median(temps)) if len(temps) > 0 else None,
                'hum_avg': float(np.mean(hums)) if len(hums) > 0 else None,
                'hum_min': float(np.min(hums)) if len(hums) > 0 else None,
                'hum_max': float(np.max(hums)) if len(hums) > 0 else None,
                'hum_median': float(np.median(hums)) if len(hums) > 0 else None
            })
    
    return hourly_stats
    
# 离群值处理函数
def handle_outliers(data, threshold=3.0):
    """
    使用Z-score方法识别和处理离群值
    :param data: 原始数据列表
    :param threshold: Z-score阈值，默认3.0
    :return: 处理后的数据列表
    """
    if data is None or len(data) == 0:  # 修改判断条件
        return data if isinstance(data, list) else []
        
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)
    
    # 计算Z-score
    z_scores = np.abs((data - mean) / std)
    
    # 识别离群值
    outliers = z_scores > threshold
    
    # 用中位数替换离群值
    median = np.median(data)
    data[outliers] = median
    
    return data.tolist()

# 数据平滑函数
def smooth_data(data, window_size=5):
    """
    使用移动平均法平滑数据
    :param data: 原始数据列表
    :param window_size: 窗口大小，默认5
    :return: 平滑后的数据列表
    """
    if not data or len(data) < window_size:
        return data
        
    window = np.ones(window_size) / window_size
    smoothed = np.convolve(data, window, mode='same')
    
    # 处理边界效应
    half_window = window_size // 2
    smoothed[:half_window] = data[:half_window]
    smoothed[-half_window:] = data[-half_window:]
    
    return smoothed.tolist()

# 新增预测相关函数
def train_predict_model(data_type='temp', look_back=None):
    """
    训练预测模型(统一使用ARIMA)
    :param data_type: 数据类型 'temp'或'hum'
    :param look_back: 使用多少历史数据点进行训练
    :return: 训练好的模型
    """
    db = get_db()
    curs = db.cursor()
    curs.execute(f"SELECT {data_type} FROM DHT_data ORDER BY timestamp DESC LIMIT 1000") # 1000个数据点
    data = np.array(curs.fetchall()).flatten()
     # 获取数据采集频率
    freq = getLastFreq() / 60  # 分钟/次

    # data = data.astype(float)
    # data = np.log1p(data)  # 使用log(1+x)避免零值问题

    # 自动计算合适的look_back
    if look_back is None:
        if freq <= 5:  # 高频采集(<=5分钟/次)
            # 10小时数据点数 = 6 * (60 / freq)
            look_back = int(10 * (60 / freq))  # 例如freq=5 → 144
        elif freq <= 60:  # 中频(<=1小时/次)
            # 10小时数据点数 = 10 * (60 / freq)
            look_back = int(10 * (60 / freq))  # 例如freq=60 → 24
        else:  # 低频(>1小时/次)
            # 7天数据点数 = 7 * 10 * (60 / freq)
            look_back = int(7 * 10 * (60 / freq))  # 例如freq=120 → 84
    # 使用auto_arima自动选择最佳参数
    model = auto_arima(
        data,
        start_p=1,      # p的最小值
        start_q=1,      # q的最小值
        max_p=5,        # p的最大值
        max_q=5,        # q的最大值
        d=1,           # 差分阶数
        seasonal=True,  # 不考虑季节性
        m=24,          # 季节性周期(24小时)
        trace=True,     # 打印搜索过程
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True   # 使用逐步算法加速
    )
    return model
# 计算评估指标
def calculate_metrics(true, pred):
    # 过滤掉None值
    true = np.array([x for x in true if x is not None])
    pred = np.array([x for x in pred if x is not None])
    
    if len(true) == 0 or len(pred) == 0:
        return {
            'MSE': 0,
            'MAE': 0,
            'MAPE': 0,
            'R2': 0
        }
    # 均方误差 (MSE)
    mse = np.mean((true - pred)**2)
    # 平均绝对误差 (MAE)
    mae = np.mean(np.abs(true - pred))
    # 平均绝对百分比误差 (MAPE)
    mape = np.mean(np.abs((true - pred)/true))*100
    # R平方
    ss_res = np.sum((true - pred)**2)
    ss_tot = np.sum((true - np.mean(true))**2)
    r2 = 1 - (ss_res/ss_tot)
    
    return {
        'MSE': float(mse),
        'MAE': float(mae),
        'MAPE': float(mape),
        'R2': float(r2)
    }
# 预测路由
@app.route('/predict', methods=['POST'])
def predict():
    data_type = request.form.get('type', 'temp')  # temp或hum

    steps = int(request.form.get('steps', 6))     # 预测步长
    model = train_predict_model(data_type)
    forecast = model.predict(n_periods=steps)

    return jsonify({
        'predictions': forecast.tolist(),
        'type': data_type,
        'steps': steps,
        'model_order': str(model.order)  # 返回模型参数
    })
# 添加定时模型更新任务
def update_models_periodically():
    with app.app_context():
        while True:
            train_predict_model('temp')
            train_predict_model('hum')
            time.sleep(3600)  # 每小时更新一次模型
# 评估路由
@app.route('/evaluate_prediction', methods=['POST'])
def evaluate_prediction():
    try:
        data_type = request.form.get('type', 'hum')  # 默认为湿度预测评估
        steps = int(request.form.get('steps', 30))    # 预测步长
        
        # 获取真实数据
        db = get_db()
        curs = db.cursor()
        curs.execute(f"SELECT {data_type} FROM DHT_data ORDER BY timestamp ASC LIMIT {steps*4}")
        data = np.array(curs.fetchall()).flatten()

        if len(data) < steps*2:
            return jsonify({'error': f'需要至少{steps*2}个历史数据点进行评估'})
        
        # 分割数据：前steps个用于训练，后steps个用于验证
        # 分割数据：前3/4用于训练，后1/4用于验证
        train_size = int(len(data) * 0.75)
        train_data = data[:train_size]
        test_data = data[train_size:train_size+steps]  # 确保测试数据长度
        # train_data = data[:steps*3]
        # test_data = data[steps*3:]
        
        # 训练模型
        model = auto_arima(
            train_data,
            start_p=0, max_p=5,
            start_q=0, max_q=5,
            d=1,
            seasonal=True,
            m=24,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        # 进行预测
        forecast = model.predict(n_periods=steps)
        # 确保预测和测试数据长度一致
        min_len = min(len(test_data), len(forecast))
        test_data = test_data[:min_len]
        forecast = forecast[:min_len]
        # 计算评估指标
        metrics = calculate_metrics(test_data, forecast)
        # 返回评估结果
        return jsonify({
            'model_order': str(model.order),        # 未使用
            'actual_values': test_data.tolist(),    # 未使用
            'predicted_values': forecast.tolist(),  # 未使用
            'metrics': metrics,
            'steps': steps                          # 未使用
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})
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
    
# 主页路由
@app.route("/")
def index():
    time, temp, hum = getLastData()
    sampleFreq = getLastFreq()  
    current_th = get_current_threshold() 
    thresholds = getThresholds() 
    selected_date1 = session.get('selected_date1', None)
    selected_date2 = session.get('selected_date2', None)
    selected_date2_1 = session.get('selected_date2_1', None) #数据统计图 第二日期
    selected_date3 = session.get('selected_date3', None)

    templateData = {
        'time': time,
        'temp': temp,
        'hum': hum, 
        'sampleFreq' : sampleFreq, 
        'current_th': current_th,
        'thresholds': thresholds,
        'selected_date1': selected_date1,
        'selected_date2': selected_date2,
        'selected_date2_1': selected_date2_1,
        'selected_date3': selected_date3
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
    selected_date1 = request.form['date']
    start_time = request.form['start_time']
    end_time = request.form['end_time']
    
    session['selected_date1'] = selected_date1
    query = """
        SELECT timestamp, temp, hum
        FROM DHT_data
        WHERE DATE(timestamp) = ?
        AND TIME(timestamp) BETWEEN ? AND ?
    """
    db = get_db()
    curs = db.cursor()
    curs.execute(query, (selected_date1, start_time, end_time))
    rows = curs.fetchall()

    templateData = {
        'data': rows

    }
    return render_template('table.html', **templateData)

#基本折线图路由
@app.route('/select_graph', methods=['POST'])
def select_graph():
    selected_date2 = request.form['date']
    selected_date2_1 = request.form['date2']
    start_time = request.form['start_time']
    end_time = request.form['end_time']

    session['selected_date2'] = selected_date2
    session['selected_date2_1'] = selected_date2_1
    # 获取当前阈值
    thresholds = getThresholds()
    query = """
        SELECT timestamp, temp, hum
        FROM DHT_data
        WHERE DATE(timestamp) = ?
        AND TIME(timestamp) BETWEEN ? AND ?
    """
    # 查询第一天的数据
    db = get_db()
    curs = db.cursor()
    curs.execute(query, (selected_date2, start_time, end_time))
    rows = curs.fetchall()
    # 查询第二天的数据
    curs.execute(query, (selected_date2_1, start_time, end_time))
    rows2 = curs.fetchall()
    # 处理数据以生成图表
    # 处理第一天的数据
    times, temps, hums = [], [], []
    for row in rows:
        times.append(row[0])
        temps.append(row[1])
        hums.append(row[2])
    # 处理第二天的数据
    times2, temps2, hums2 = [], [], []
    for row in rows2:    
        times2.append(row[0])
        temps2.append(row[1])
        hums2.append(row[2])
    
# 使用Plotly绘制温度图像
    trace_temp = go.Scatter(x=times, y=temps, mode='lines', name='温度 (°C)')

    layout_temp = go.Layout(
        title='温度随时间变化趋势',
        xaxis=dict(title='时间'),
        yaxis=dict(title='温度 (°C)', range=[20, 40]),  # 固定纵轴范围
        hovermode='x unified'
    )
# 绘制数据处理后的图像
#去除离群值
    temps_ho = handle_outliers(temps)
    hums_ho = handle_outliers(hums)
    temps2_ho = handle_outliers(temps2)
    hums2_ho = handle_outliers(hums2)
# 温度制图
    trace_temp_ho = go.Scatter(
        x=times,  # x轴数据（时间）
        y=temps_ho,  # y轴数据（温度值）
        mode='lines',
        name=f'{selected_date2} 温度 (°C)',    
    )
    layout_temp_ho = go.Layout(
        title='温度随时间变化趋势_去离群值',
        xaxis=dict(title='时间'),
        yaxis=dict(title='温度 (°C)', range=[20, 40]),  # 固定纵轴范围
        hovermode='x unified'
    )
    fig_temp_ho = go.Figure(data=[trace_temp_ho], layout=layout_temp_ho)
    plot_div_temp_ho = pyo.plot(fig_temp_ho, output_type='div', include_plotlyjs=False)
# 湿度制图
    trace_hum_ho = go.Scatter(
        x=times,  # x轴数据（时间）
        y=hums_ho,  # y轴数据（湿度值）
        mode='lines',
        name=f'{selected_date2} 湿度 (%)',    
    )
    layout_hum_ho = go.Layout(
        title='湿度随时间变化趋势_去离群值',
        xaxis=dict(title='时间'),
        yaxis=dict(title='湿度 (%)', range=[0, 100]),  # 固定纵轴范围
        hovermode='x unified'
    )
    fig_hum_ho = go.Figure(data=[trace_hum_ho], layout=layout_hum_ho)
    plot_div_hum_ho = pyo.plot(fig_hum_ho, output_type='div', include_plotlyjs=False)
#数据平滑
    temps_sm = smooth_data(temps_ho, window_size=5)
    hums_sm = smooth_data(hums_ho, window_size=5)
    # 制图
    trace_temp_sm = go.Scatter(
        x=times,  # x轴数据（时间）
        y=temps_sm,  # y轴数据（温度值）
        mode='lines',
        name=f'{selected_date2} 温度 (°C)',    
    )
    layout_temp_sm = go.Layout(
        title='温度随时间变化趋势_平滑',
        xaxis=dict(title='时间'),
        yaxis=dict(title='温度 (°C)', range=[20, 40]),  # 固定纵轴范围
        hovermode='x unified'
    )
    fig_temp_sm = go.Figure(data=[trace_temp_sm], layout=layout_temp_sm)
    plot_div_temp_sm = pyo.plot(fig_temp_sm, output_type='div', include_plotlyjs=False)
    # 湿度制图
    trace_hum_sm = go.Scatter(
        x=times,  # x轴数据（时间）
        y=hums_sm,  # y轴数据（湿度值）
        mode='lines',
        name=f'{selected_date2} 湿度 (%)',    
    )
    layout_hum_sm = go.Layout(
        title='湿度随时间变化趋势_平滑',
        xaxis=dict(title='时间'),
        yaxis=dict(title='湿度 (%)', range=[0, 100]),  # 固定纵轴范围
        hovermode='x unified'
    )
    fig_hum_sm = go.Figure(data=[trace_hum_sm], layout=layout_hum_sm)
    plot_div_hum_sm = pyo.plot(fig_hum_sm, output_type='div', include_plotlyjs=False)
# 添加温度阈值线
    temp_thresholds = []
    for th in thresholds:
    # 根据时段设置不同颜色
        color = 'red' if th[0] == 'A' else 'green' if th[0] == 'B' else 'blue'
        temp_thresholds.append(go.Scatter(
            x=[times[0], times[-1]], 
            y=[th[3], th[3]], 
            mode='lines', 
            name=f'{th[0]}时段温度阈值', 
            line=dict(color=color, dash='dash')
        ))
    fig_temp = go.Figure(data=[trace_temp] + temp_thresholds, layout=layout_temp)
    plot_div_temp = pyo.plot(fig_temp, output_type='div', include_plotlyjs=False)
 
# 使用Plotly绘制湿度图像
    trace_hum = go.Scatter(x=times, y=hums, mode='lines', name='湿度 (%)')
    layout_hum = go.Layout(
        title='湿度随时间变化趋势',
        xaxis=dict(title='时间'),
        yaxis=dict(title='湿度 (%)', range=[0, 100]),  # 固定纵轴范围
        hovermode='x unified'
    )
# 添加湿度阈值线
    hum_thresholds = []
    for th in thresholds:
        # 根据时段设置不同颜色
        color = 'red' if th[0] == 'A' else 'green' if th[0] == 'B' else 'blue'
        hum_thresholds.append(go.Scatter(
            x=[times[0], times[-1]], 
            y=[th[4], th[4]], 
            mode='lines', 
            name=f'{th[0]}时段湿度阈值', 
            line=dict(color=color, dash='dash')
        ))
    fig_hum = go.Figure(data=[trace_hum] + hum_thresholds, layout=layout_hum)
    plot_div_hum = pyo.plot(fig_hum, output_type='div', include_plotlyjs=False)
# 使用Plotly绘制温湿度散点图
    trace_temp_hum = go.Scatter(
        x=temps, 
        y=hums, 
        mode='markers', 
        name='温湿度散点图',
        marker=dict(size=5, color='blue', opacity=0.5)
    )
    layout_temp_hum = go.Layout(
        title='温湿度散点图',
        xaxis=dict(title='温度 (°C)'),
        yaxis=dict(title='湿度 (%)'),
        hovermode='closest'
    )
    fig_temp_hum = go.Figure(data=[trace_temp_hum], layout=layout_temp_hum)
    plot_div_temp_hum = pyo.plot(fig_temp_hum, output_type='div', include_plotlyjs=False)

# 计算一阶差分
    temp_diff = [0] + [temps[i] - temps[i-1] for i in range(1, len(temps))]
    hum_diff = [0] + [hums[i] - hums[i-1] for i in range(1, len(hums))]

# 使用Plotly绘制温度差分图像
    trace_temp_diff = go.Scatter(x=times, y=temp_diff, mode='lines', name='温度变化 (°C)')
    layout_temp_diff = go.Layout(
        title='一阶差分·温度变化量',
        xaxis=dict(title='时间'),
        yaxis=dict(
        title='温度变化 (°C)',
        range=[-5, 5],  # 强制y轴范围在-5到5之间
        autorange=False  # 禁用自动缩放
    ),
        hovermode='x unified'
    )
    fig_temp_diff = go.Figure(data=[trace_temp_diff], layout=layout_temp_diff)
    plot_div_temp_diff = pyo.plot(fig_temp_diff, output_type='div', include_plotlyjs=False)

    # 使用Plotly绘制湿度差分图像
    trace_hum_diff = go.Scatter(x=times, y=hum_diff, mode='lines', name='湿度变化 (%)')
    layout_hum_diff = go.Layout(
        title='一阶差分·湿度变化量',
        xaxis=dict(title='时间'),
        yaxis=dict(
        title='湿度变化 (%)',
        range=[-10, 10],  # 强制y轴范围在-15到15之间
        autorange=False  # 禁用自动缩放
    ),
        hovermode='x unified'
    )
    fig_hum_diff = go.Figure(data=[trace_hum_diff], layout=layout_hum_diff)
    plot_div_hum_diff = pyo.plot(fig_hum_diff, output_type='div', include_plotlyjs=False)
     # 使用Plotly绘制温度对比图像
    trace_temp = go.Scatter(
    x=list(range(len(temps_ho))),  # x轴数据（索引）
    y=temps_ho,              # y轴数据（温度值）
    mode='lines',
    name=f'{selected_date2} 温度 (°C)',
    hovertemplate='时间: %{text|%H:%M}<br>温度: %{y}°C<extra></extra>',  # 自定义悬停模板
    text=times            # 将时间数据绑定到text属性，供hovertemplate调用
)
    trace_temp2 = go.Scatter(x=list(range(len(temps2_ho))),
    y=temps2_ho,
    mode='lines', 
    name=f'{selected_date2_1} 温度 (°C)',
    hovertemplate='时间: %{text|%H:%M}<br>湿度: %{y}°C<extra></extra>',  # 自定义悬停模板
    text=times2            # 将时间数据绑定到text属性，供hovertemplate调用
    )
    layout_temp_compare = go.Layout(
        title='温度对比图',
        xaxis=dict(title='样本'),
        yaxis=dict(title='温度 (°C)'),
        hovermode='x unified'
    )
    fig_temp_compare = go.Figure(data=[trace_temp, trace_temp2], layout=layout_temp_compare)
    plot_div_temp_compare = pyo.plot(fig_temp_compare, output_type='div', include_plotlyjs=False)

    # 使用Plotly绘制湿度对比图像
    trace_hum = go.Scatter(
    x=list(range(len(hums_ho))),  # x轴数据（索引）
    y=hums_ho,              # y轴数据（湿度值）
    mode='lines',
    name=f'{selected_date2} 湿度 (%)',
    hovertemplate='时间: %{text|%H:%M}<br>湿度: %{y}%<extra></extra>',  # 自定义悬停模板
    text=times            # 将时间数据绑定到text属性，供hovertemplate调用
)
    trace_hum2 = go.Scatter(
    x=list(range(len(hums2_ho))),  # x轴数据（索引）
    y=hums2_ho, 
    mode='lines', 
    name=f'{selected_date2_1} 湿度 (%)',
    hovertemplate='时间: %{text|%H:%M}<br>湿度: %{y}%<extra></extra>',  # 自定义悬停模板
    text=times2            # 将时间数据绑定到text属性，供hovertemplate调用
)
    layout_hum_compare = go.Layout(
        title='湿度对比图',
        xaxis=dict(title='样本'),
        yaxis=dict(title='湿度 (%)'),
        hovermode='x unified'
    )
    fig_hum_compare = go.Figure(data=[trace_hum, trace_hum2], layout=layout_hum_compare)
    plot_div_hum_compare = pyo.plot(fig_hum_compare, output_type='div', include_plotlyjs=False)
    
    templateData = {
        'times': times,
        'temps': temps,
        'hums': hums, 
        'start_time': start_time,
        'end_time': end_time,
        'selected_date2': selected_date2,
        'plot_div_temp': plot_div_temp, # Plotly温度图像的HTML代码
        'plot_div_hum': plot_div_hum,  # Plotly湿度图像的HTML代码
        'plot_div_temp_ho': plot_div_temp_ho, # 温度去离群值图像
        'plot_div_hum_ho': plot_div_hum_ho, # 湿度去离群值图像
        'plot_div_temp_sm': plot_div_temp_sm, # 温度平滑图像
        'plot_div_hum_sm': plot_div_hum_sm, # 湿度平滑图像
        'plot_div_temp_hum': plot_div_temp_hum, # 温湿度散点图像
        'plot_div_temp_diff': plot_div_temp_diff,  # 温度差分图像
        'plot_div_hum_diff': plot_div_hum_diff,  # 湿度差分图像
        'plot_div_temp_compare': plot_div_temp_compare,  # 温度对比图
        'plot_div_hum_compare': plot_div_hum_compare  # 湿度对比图
    }
    return render_template('graphs.html', **templateData)

# 数据统计计算路由
@app.route('/calculate_stats', methods=['POST'])
def handle_calculate_stats():
    selected_date = request.form['stats_date']
    start_time = request.form['stats_start']
    end_time = request.form['stats_end']
    
    stats = calculate_statistics(selected_date, start_time, end_time)
    
    if not stats:
        return jsonify({'error': '没有找到指定时段的数据'})
    
    return jsonify(stats)

# 数据统计保存路由
@app.route('/save_stats', methods=['POST'])
def handle_save_stats():
    try:
        selected_date = request.form['stats_date']
        start_time = request.form['stats_start']
        end_time = request.form['stats_end']
        
        stats = calculate_statistics(selected_date, start_time, end_time)
        if not stats:
            return jsonify({'error': '没有找到指定时段的数据'})
            
        save_stats_result(selected_date, selected_date, start_time, end_time, stats)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)})

# 数据统计历史查询路由
@app.route('/query_stats_history', methods=['POST'])
def query_stats_history():
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    
    db = get_db()
    curs = db.cursor()
    curs.execute('''
        SELECT id, start_date, start_time, end_time, 
               temp_avg, hum_avg, correlation
        FROM stats_results
        WHERE start_date BETWEEN ? AND ?
        ORDER BY start_date DESC, start_time DESC
    ''', (start_date, end_date))
    
    rows = curs.fetchall()
    if not rows:
        return jsonify({'error': '没有找到指定日期范围内的记录'})
    
    # 转换为字典列表
    results = []
    for row in rows:
        results.append({
            'id': row[0],
            'start_date': row[1],
            'start_time': row[2],
            'end_time': row[3],
            'temp_avg': row[4],
            'hum_avg': row[5],
            'correlation': row[6]
        })
    
    return jsonify(results)
# 数据统计详情查询路由
@app.route('/get_stats_detail')
def get_stats_detail():
    record_id = request.args.get('id')
    
    db = get_db()
    curs = db.cursor()
    curs.execute('''
        SELECT * FROM stats_results WHERE id = ?
    ''', (record_id,))
    
    row = curs.fetchone()
    if not row:
        return jsonify({'error': '记录不存在'})
    
    # 解析JSON格式的区间数据
    
    # 修复JSON解析问题
    def safe_json_loads(json_str):
        try:
            return json.loads(json_str.replace("'", '"'))  # 将单引号替换为双引号
        except:
            return {}
    temp_bins = safe_json_loads(row[14]) if row[14] else {}
    hum_bins = safe_json_loads(row[15]) if row[14] else {}
    
    return jsonify({
        'id': row[0],
        'start_date': row[1],
        'end_date': row[2],
        'start_time': row[3],
        'end_time': row[4],
        'temp_avg': row[5],
        'temp_min': row[6],
        'temp_max': row[7],
        'temp_median': row[8],
        'hum_avg': row[9],
        'hum_min': row[10],
        'hum_max': row[11],
        'hum_median': row[12],
        'correlation': row[13],
        'temp_bins': temp_bins,
        'hum_bins': hum_bins
    })

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
    current_th = get_current_threshold()
    thresholds = getThresholds()
    time, temp, hum = getLastData()
    templateData = {
        'time': time,
        'temp': temp,
        'hum': hum,
        'current_th': current_th,
        'thresholds': thresholds,
        'sampleFreq': sampleFreq   ###
    }
    return render_template('index.html', **templateData)

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

# 报警历史查询路由
@app.route('/query_alarm_history', methods=['POST'])
def query_alarm_history():
    selected_date3 = request.form['date']
    start_time = request.form['start_time']
    end_time = request.form['end_time']
    session['selected_date3'] = selected_date3
    
    db = get_db()
    curs = db.cursor()
    query = """
        SELECT timestamp, alarm_type, current_value, threshold 
        FROM alarm_logs
        WHERE DATE(timestamp) = ?
        AND TIME(timestamp) BETWEEN ? AND ?
        ORDER BY timestamp DESC
    """
    curs.execute(query, (selected_date3, start_time, end_time))
    rows = curs.fetchall()

    templateData = {
        'alarm_data': rows
        # 'selected_date': selected_date,
        # 'start_time': start_time,
        # 'end_time': end_time
    }
    return render_template('table_alarm.html', **templateData)

if __name__ == "__main__":
    # 启动后台监控
    monitor_thread = Thread(target=background_monitor, daemon=True)
    monitor_thread.start() 
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

