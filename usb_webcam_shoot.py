import cv2
import time

# Input camera stream params
input_width = 640
input_height = 480
input_fps = 30

# Input format
input_format = 'jpeg'  # or 'h264'
output_filename_template = "webcam{}.jpg"

# GStreamer pipeline setup
if input_format == 'jpeg':
    pipeline = f"v4l2src device=/dev/video0 ! image/jpeg, width={input_width}, height={input_height}, framerate={input_fps}/1 ! jpegparse ! jpegdec ! videoconvert ! appsink sync=false drop=true"
elif input_format == 'h264':
    pipeline = f"v4l2src device=/dev/video0 io-mode=2 ! video/x-h264, width={input_width}, height={input_height}, framerate={input_fps}/1 ! h264parse ! avdec_h264 ! videoconvert ! appsink sync=false drop=true"

print("Input video pipeline:", pipeline)

# Video capture initialization
video = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

# Image capture loop
index = 0
while True:
    # Read the video frames
    ret, frame = video.read()

    if not ret:
        break
    #resized_frame = cv2.resize(frame,  (640, 480)) 
    
        # Crop the frame to make it 480x480 by taking the center region
    start_point = int((input_width - input_height) / 2)
    end_point = int((input_width + input_height) / 2)
    cropped_frame = frame[:, start_point:end_point, :]
    
    
    # Display live preview
    cv2.imshow("Live Preview", cropped_frame)

    # Wait for the 'Enter' key (key code 13) or 'Esc' key (key code 27)
    key = cv2.waitKey(1)
    if key == 13:  # Enter key pressed
        # Determine the filename
        filename = output_filename_template.format(time.strftime("%Y%m%d-%H%M%S"))
        index += 1

        # Save the frame as an image
        cv2.imwrite(filename, frame)
        print(f"Image saved as {filename}")

    elif key == 27:  # Esc key pressed
        break

# Release video source
video.release()
cv2.destroyAllWindows()
