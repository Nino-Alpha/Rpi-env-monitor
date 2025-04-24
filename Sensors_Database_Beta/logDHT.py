import time
import board
import sqlite3
import adafruit_dht
import os
os.environ["BLINKA_FORCEBOARD"] = "RASPBERRY_PI_4B"  # 强制指定树莓派 4B
dbname='sensorsData.db'
print("Board attributes:", dir(board))
print(board.__file__)  # 打印 board 模块的文件路径

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
		displayData()   #输出监控。
		time.sleep(sampleFreq)


# if __name__ == '__main__':
# 	main()
main()

