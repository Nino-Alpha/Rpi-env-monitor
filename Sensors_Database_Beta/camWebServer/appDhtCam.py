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
#from Sensors_Database_Beta import logDHT

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

def getLastData():
    db = get_db()
    curs = db.cursor()
    curs.execute("SELECT * FROM DHT_data ORDER BY timestamp DESC LIMIT 1")
    row = curs.fetchone()
    if row:
        time, temp, hum = str(row[0]), row[1], row[2]
        return time, temp, hum
    return None, None, None

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

def getThresholds():
    db = get_db()
    curs = db.cursor()
    curs.execute("SELECT * FROM thresholds ORDER BY period")
    return curs.fetchall()

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
    
    # 基础统计
    stats = {
        'temp_avg': np.mean(temps),
        'temp_min': np.min(temps),
        'temp_max': np.max(temps),
        'temp_median': np.median(temps),
        'hum_avg': np.mean(hums),
        'hum_min': np.min(hums),
        'hum_max': np.max(hums),
        'hum_median': np.median(hums),
        'correlation': np.corrcoef(temps, hums)[0, 1],  
    }
    # 温度区间百分比
    temp_bins = [0, 10, 20, 30, 40, 50]
    temp_counts, _ = np.histogram(temps, bins=temp_bins)
    stats['temp_bins'] = {
        f'{temp_bins[i]}-{temp_bins[i+1]}': count/len(temps)*100 
        for i, count in enumerate(temp_counts)
    }
    
    # 湿度区间百分比
    hum_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    hum_counts, _ = np.histogram(hums, bins=hum_bins)
    stats['hum_bins'] = {
        f'{hum_bins[i]}-{hum_bins[i+1]}': count/len(hums)*100 
        for i, count in enumerate(hum_counts)
    }
    hourly_stats = calculate_hourly_statistics(selected_date, start_time, end_time) # 计算每小时统计数据
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
            
        temps = data[:, 0]
        hums = data[:, 1]
        
        hourly_stats.append({
            'time_range': f"{hour_start}-{hour_end}",
            'temp_avg': np.mean(temps),
            'temp_min': np.min(temps),
            'temp_max': np.max(temps),
            'temp_median': np.median(temps),
            'hum_avg': np.mean(hums),
            'hum_min': np.min(hums),
            'hum_max': np.max(hums),
            'hum_median': np.median(hums)
        })
    
    return hourly_stats

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

#数据库统计图路由
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
    times, times1, temps, hums = [], [], [], []
    for row in rows:

        times.append(row[0])
        temps.append(row[1])
        hums.append(row[2])
        # # 只保留时间部分time1用于
        # time_str = row[0].split(' ')[1]  # 提取时间部分
        # times1.append(time_str)
    # 处理第二天的数据
    times2, temps2, hums2 = [], [], []
    for row in rows2:
         # 只保留时间部分
        # time_str = row[0].split(' ')[1]  # 提取时间部分
        # times2.append(time_str)
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
    # 使用Plotly绘制温湿度图像
    layout = go.Layout(
    title='温湿度随时间变化趋势',
    xaxis=dict(title='时间'),
    yaxis=dict(title='数值'),
    hovermode='x unified'
    )
    fig_combined = go.Figure(data=[trace_temp, trace_hum], layout=layout)
    plot_div_combined = pyo.plot(fig_combined, output_type='div', include_plotlyjs=False)
    # 计算一阶差分
    temp_diff = [0] + [temps[i] - temps[i-1] for i in range(1, len(temps))]
    hum_diff = [0] + [hums[i] - hums[i-1] for i in range(1, len(hums))]

    # 使用Plotly绘制温度差分图像
    trace_temp_diff = go.Scatter(x=times, y=temp_diff, mode='lines', name='温度变化 (°C)')
    layout_temp_diff = go.Layout(
        title='一阶差分·温度变化量',
        xaxis=dict(title='时间'),
        yaxis=dict(title='温度变化 (°C)'),
        hovermode='x unified'
    )
    fig_temp_diff = go.Figure(data=[trace_temp_diff], layout=layout_temp_diff)
    plot_div_temp_diff = pyo.plot(fig_temp_diff, output_type='div', include_plotlyjs=False)

    # 使用Plotly绘制湿度差分图像
    trace_hum_diff = go.Scatter(x=times, y=hum_diff, mode='lines', name='湿度变化 (%)')
    layout_hum_diff = go.Layout(
        title='一阶差分·湿度变化量',
        xaxis=dict(title='时间'),
        yaxis=dict(title='湿度变化 (%)'),
        hovermode='x unified'
    )
    fig_hum_diff = go.Figure(data=[trace_hum_diff], layout=layout_hum_diff)
    plot_div_hum_diff = pyo.plot(fig_hum_diff, output_type='div', include_plotlyjs=False)
     # 使用Plotly绘制温度对比图像
    trace_temp = go.Scatter(
    x=list(range(len(temps))),  # x轴数据（索引）
    y=temps,              # y轴数据（温度值）
    mode='lines',
    name=f'{selected_date2} 温度 (°C)',
    hovertemplate='时间: %{text|%H:%M}<br>温度: %{y}°C<extra></extra>',  # 自定义悬停模板
    text=times            # 将时间数据绑定到text属性，供hovertemplate调用
)
    trace_temp2 = go.Scatter(x=list(range(len(temps2))),
    y=temps2,
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
    x=list(range(len(hums))),  # x轴数据（索引）
    y=hums,              # y轴数据（湿度值）
    mode='lines',
    name=f'{selected_date2} 湿度 (%)',
    hovertemplate='时间: %{text|%H:%M}<br>湿度: %{y}%<extra></extra>',  # 自定义悬停模板
    text=times            # 将时间数据绑定到text属性，供hovertemplate调用
)
    trace_hum2 = go.Scatter(
    x=list(range(len(hums2))),  # x轴数据（索引）
    y=hums2, 
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
        # 'plot_url_temp': plot_url_temp,  # 温度图像URL
        'plot_div_temp': plot_div_temp, # Plotly温度图像的HTML代码
        # 'plot_url_hum': plot_url_hum ,   # 湿度图像URL
        'plot_div_hum': plot_div_hum,  # Plotly湿度图像的HTML代码
        'plot_div_combined': plot_div_combined , # 温湿度组合图表
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
    # return jsonify({
    #     'temp_avg': stats['temp_avg'],
    #     'temp_min': stats['temp_min'],
    #     'temp_max': stats['temp_max'],
    #     'temp_median': stats['temp_median'],
    #     'hum_avg': stats['hum_avg'],
    #     'hum_min': stats['hum_min'],
    #     'hum_max': stats['hum_max'],
    #     'hum_median': stats['hum_median'],
    #     'correlation': stats['correlation'],
    #     'hourly_stats': stats['hourly_stats'],
    #     'temp_pie': stats['temp_pie'],
    #     'hum_pie': stats['hum_pie']
    # })

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
    time, temp, hum = getLastData()
    templateData = {
        'time': time,
        'temp': temp,
        'hum': hum,
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