import time
import io
import threading
from picamera2 import Picamera2
import sys
sys.path.append('/usr/lib/python3/dist-packages')
from libcamera import Transform

class Camera(object):
    thread = None  # background thread that reads frames from camera
    frame = None  # current frame is stored here by background thread
    last_access = 0  # time of last client access to the camera

    def initialize(self):
        if Camera.thread is None:
            # start background frame thread
            Camera.thread = threading.Thread(target=self._thread)
            Camera.thread.start()

            # wait until frames start to be available
            while self.frame is None:
                time.sleep(0)

    def get_frame(self):
        Camera.last_access = time.time()
        self.initialize()
        return self.frame

    @classmethod
    def _thread(cls):
        # Initialize Picamera2
        picam2 = Picamera2()

        # Create configuration with Transform for flip settings
        transform = Transform(hflip=True, vflip=True)
        config = picam2.create_video_configuration(main={"size": (320, 240)}, transform=transform)
        picam2.configure(config)

        # Start the camera
        picam2.start()

        # Warm up the camera
        time.sleep(2)

        stream = io.BytesIO()
        while True:
            # Capture frame
            picam2.capture_file(stream, format="jpeg")
            stream.seek(0)
            cls.frame = stream.read()

            # Reset stream for next frame
            stream.seek(0)
            stream.truncate()

            # Check if thread should stop
            if time.time() - cls.last_access > 10:
                break

        # Clean up
        picam2.stop()
        cls.thread = None