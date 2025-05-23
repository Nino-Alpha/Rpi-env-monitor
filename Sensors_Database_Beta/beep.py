import time 
import RPi.GPIO as GPIO
# 设置 编号模式


GPIO.setmode(GPIO.BOARD)  # 设置 GPIO 引脚编号模式为 board
#
print ("RPi.GPIO version:", GPIO.VERSION)  # 打印 RPi.GPIO 版本
#
GPIO.setup(13, GPIO.OUT)  # 设置 GPIO 13 引脚为输出模式
def beep(seconds):

	GPIO.output(13, GPIO.LOW)  # 设置 GPIO 11 引脚为低电平
	time.sleep(seconds)  # 延时 * 秒,即响应时间
	GPIO.output(13, GPIO.HIGH)  # 设置 GPIO 11 引脚为高电平

def beepAction(secs,sleepsecs,times):
	for i in range(times): # 循环次数,代表总时长
		beep(secs)			# 鸣叫时间时间
		time.sleep(sleepsecs)  # 停顿间隔
	print ("beepAction finished")
		

beepAction(0.02,1,5) # 鸣叫时间0.02秒，停顿间隔0.2秒，循环5次
GPIO.cleanup()  # 清理 GPIO 设置