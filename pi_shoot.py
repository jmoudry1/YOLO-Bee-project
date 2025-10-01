import time
import cv2
from picamera2 import Picamera2
from PIL import Image
from io import BytesIO

# Camera parameters
input_width = 640
input_height = 480

#################################################################
# IMPORTANT! Define coordinates for cropping
# You need crop picture to [almost] perfect square 380x380
#change this
x1, y1 = 120, 120  # Default (no crop) coordinates 
x2, y2 = 480, 480  # Default (no crop) coordinates 
#with this
#x1, y1 = 145, 60  # Real crop coordinates (adjust as needed)
#x2, y2 = 525, 440  # Real coordinates (adjust as needed)
################################################################

# Output filename template
output_filename_template = "test{}.jpg"

# Camera setup
tuning = Picamera2.load_tuning_file("/usr/share/libcamera/ipa/rpi/pisp/imx477.json")
picam2 = Picamera2(tuning=tuning)
main = {'size': (input_width, input_height), 'format': 'RGB888'}
controls = {
    'FrameRate': 30,
    'AnalogueGain': 50,
    'AwbEnable': True,
    'AeEnable': False,
    'Brightness': 0,
    'Contrast': 1,
    'Saturation': 1,
    'ColourGains': (3.4, 1.5)
}
config = picam2.create_video_configuration(main, controls=controls)
picam2.configure(config)

picam2.start()

# Video preview loop
try:
    while True:
        # Capture frame
        frame = picam2.capture_array()

        # Crop the frame using the specified coordinates
        cropped_frame = frame[y1:y2, x1:x2, :]

        # Display the frame
        cv2.imshow("Video Preview", cropped_frame)

        # Check for key press
        key = cv2.waitKey(1)
        if key == 13:  # Enter key
            # Capture image
            output_filename = output_filename_template.format(time.strftime("%Y%m%d-%H%M%S"))
            cv2.imwrite(output_filename, cropped_frame)
            print(f"Captured: {output_filename}")

        elif key == 27:  # Esc key
            break

finally:
    # Release resources
    picam2.stop()
    cv2.destroyAllWindows()
