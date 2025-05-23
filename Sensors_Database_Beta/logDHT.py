import time
import board
import sqlite3
import adafruit_dht
import os
import RPi.GPIO as GPIO
import time 
from datetime import datetime 
# 设置 编号模式
GPIO.setmode(GPIO.BCM)  # 设置 GPIO 引脚编号模式为 BCM，与树莓派引脚编号一致

GPIO.setup(27, GPIO.OUT)  # 设置 GPIO 27 引脚为输出模式

os.environ["BLINKA_FORCEBOARD"] = "RASPBERRY_PI_4B"  # 强制指定树莓派 4B
dbname='sensorsData.db'
# print("Board attributes:", dir(board))
# print(board.__file__)  # 打印 board 模块的文件路径
def beep(seconds):

	GPIO.output(27, GPIO.LOW)  # 设置 GPIO 27 引脚为低电平
	time.sleep(seconds)  # 延时 * 秒,即响应时间
	GPIO.output(27, GPIO.HIGH)  # 设置 GPIO 27 引脚为高电平

def beepAction(secs,sleepsecs,times):
	for i in range(times): # 循环次数,代表总时长
		beep(secs)			# 鸣叫时间时间
		time.sleep(sleepsecs)  # 停顿间隔
	GPIO.cleanup()  # 清理 GPIO 设置

def get_current_threshold():
    current_time = datetime.now().strftime("%H:%M")
    conn = sqlite3.connect(dbname)
    curs = conn.cursor()
    curs.execute('''SELECT * FROM thresholds 
        WHERE start_time <= ? AND end_time >= ? 
        ORDER BY period LIMIT 1''', (current_time, current_time))
    row = curs.fetchone()
    conn.close()
    return row

def getDHTdata(retries=5):
    dht = adafruit_dht.DHT11(board.D17, use_pulseio=False)  # GPIO 17 引脚
    try:
        for _ in range(retries):
            try:
                # 读取温湿度数据
                temp = dht.temperature
                hum = dht.humidity + 13 # 湿度值加13，校正值
                if hum is not None and temp is not None:
                    hum = round(hum)
                    temp = round(temp, 1)
                    return temp, hum
                else:
                    print("Failed to read sensor data. Retrying...")
            except RuntimeError as e:
                print(f"Error reading sensor: {e}. Retrying...")
            time.sleep(2)  # 等待 2 秒后重试

        print("Max retries reached. Failed to read sensor data.")
        return None, None
    finally:
        # 清理传感器资源
        dht.exit()

def logData (temp, hum):
	
	conn=sqlite3.connect(dbname)
	curs=conn.cursor()	
	curs.execute("INSERT INTO DHT_data values(datetime('now','localtime'), (?), (?))", (temp, hum)) # modify : localtime
	conn.commit()
	conn.close()
	
def getFreqdata():	
	conn=sqlite3.connect(dbname)
	curs=conn.cursor()
	curs.execute("SELECT * FROM Sample_freq ORDER BY timestamp DESC LIMIT 1")
	row = curs.fetchone()
	if row:
		freq = row[1]
		return freq
	return None

def displayData():
	conn=sqlite3.connect('sensorsData.db')
	curs=conn.cursor()
	print ("\nLast DHT Data logged on database:\n")  
	for row in curs.execute("SELECT * FROM DHT_data ORDER BY timestamp DESC LIMIT 1"):
		print (str(row[0])+" ==> Temp = "+str(row[1])+"	Hum ="+str(row[2]))

	print ("\nLast Freq Data logged on database:\n")
	for row in curs.execute("SELECT * FROM Sample_freq ORDER BY timestamp DESC LIMIT 1"):
		print (str(row[0])+" ==> Freq = "+str(row[1]))

def main():
	while True:
		sampleFreq = getFreqdata()
		temp, hum = getDHTdata()
		logData (temp, hum)
		row = get_current_threshold()
		if row:
			threshold_temp = row[3]
			threshold_hum = row[4]
			temp_exceed = temp > int(threshold_temp)
			hum_exceed = hum > int(threshold_hum)

			if temp_exceed and hum_exceed:  # 情况3: 温湿度同时超标
				print("Temperature and Humidity both exceed threshold!")
				beepAction(0.02, 0.1, 10)  # 急促连续鸣叫
			elif temp_exceed:  # 情况1: 仅温度超标
				print("Temperature exceeds threshold!")
				beepAction(0.02, 0.2, 5)  # 短间隔、多鸣叫
			elif hum_exceed:  # 情况2: 仅湿度超标
				print("Humidity exceeds threshold!")
				beepAction(0.02, 1, 3)    # 长间隔、少鸣叫
			else:
				GPIO.output(27, GPIO.HIGH)  # 静音状态
				print("Temp/Hum is normal.")
		displayData()   #输出监控。
		time.sleep(sampleFreq)


if __name__ == '__main__':
	main()


