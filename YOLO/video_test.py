import cv2
import time
import numpy as np
import gpiod
from gpiod.line import Direction, Value
from ultralytics import YOLO
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder, JpegEncoder, Quality

#****************************************************************************************************************************
# SETTINGS SECTION
#****************************************************************************************************************************


# input camera stream params
input_width = 640
input_height = 480
input_fps = 30 # input fps has appropriate values depending on resolution. It can be 10, 15, 20, 30 or 60 fps
input_format = 'h264' #  jpeg or h264 mode for camera
frame_number =0


# detector params
model = YOLO("zomb320.onnx")  # Replace with the actual path to your YOLO model (filename or dir for NCNN)
model_width = 320   # as it was trained or converted
model_height = 320  # as it was trained or converted
model_confidence = 0.75  
model_iou = 0.2 # overlap of two detected objects
model_tracker = 'botsort.yaml'  # botsort.yaml (default) or bytetrack.yaml. Used for object tracking (model.track method)
model_type = 'ONNX' # for stats on video, not affects anything important

# output video params

# Define coordinates for cropping
crop_x1, crop_y1 = 120, 120  # Example coordinates (adjust as needed)
crop_x2, crop_y2 = 480, 480  # Example coordinates (adjust as needed)
output_width = 360
output_height = 360
output_fps = 20
output_fourcc = "mp4v"
output_filename = "video_test.mp4"
osd_status = 'LAUNCHED'  # On-screen status to put on video out

# motion detector params
cell_size = 10 # Define cell size as 20x20 pixels (adjust as needed)
background_threshold = 50  # Adjust the threshold between background and frame in 0..255 (grayscale)
min_area = 100 #the minimum size (in pixels) for a region of an image to be considered actual “motion” (20x20)

#background definition
background_image = cv2.imread("back.jpg") 
background_frame = background_image  # used after turbine comes from on to off state
background_image = cv2.cvtColor(background_image, cv2.COLOR_RGB2GRAY)


#relay params
CH1_LINE = 26  #relay 1. Currently only this in use
CH2_LINE = 20  #relay 2
CH3_LINE = 21  #relay 3
#turbine_working_interval_ms = 3000 # 3 seconds to switch relay ON




#*******************************************************************************************************************************
# INIT SECTION
#*******************************************************************************************************************************
print ("Model params: [" + str(model_width) + "," + str(model_height) + "], conf=" + str(model_confidence) + ", iou=" + str(model_iou) + ", tracker=" + model_tracker)
#camera init
#-----------------------------------------------------------------------------------------
tuning = Picamera2.load_tuning_file("/usr/share/libcamera/ipa/rpi/pisp/imx477.json")
picam2 = Picamera2(tuning=tuning)

main = {'size': (input_width, input_height), 'format': 'RGB888'} 
controls = {
'FrameRate': input_fps,
'AnalogueGain' : 50, 
#'ExposureTime' : 100000,
'AwbEnable' : True,
'AeEnable' : False,
'Brightness' : 0, # -1 to 1
'Contrast' : 1,  # 0 to 32 float, normal around 1
'Saturation' : 1, # 0 to 32 float, normal around 1
'ColourGains': (3.4, 1.5) # red, blue

}   
config = picam2.create_video_configuration(main, controls=controls)
picam2.configure(config)

print (config)
picam2.start()


#if input_format == 'jpeg':
#    encoder = JpegEncoder(q=70)

#if input_format == 'h264':
#	encoder = H264Encoder(1000000)
	

fourcc = cv2.VideoWriter_fourcc(*output_fourcc)  
output_video = cv2.VideoWriter(output_filename, fourcc, output_fps, (output_width, output_height))
#-------------------------------------------------------------------------------------------------

# Artifact detector init
# Function to split the image into cells and calculate average color of each cell
def calculate_average_intensity(image, cell_size):
    height, width = image.shape
    average_intensities = []

    for y in range(0, height, cell_size):
        for x in range(0, width, cell_size):
            cell = image[y:y+cell_size, x:x+cell_size]
            average_intensity = np.mean(cell)
            average_intensities.append(average_intensity)

    return np.array(average_intensities)
background_avg_colors = calculate_average_intensity(background_image, cell_size)
#-------------------------------------------------------------------------------------------------

#Relay init
def initialize_gpio():
    lines = gpiod.request_lines(
    "/dev/gpiochip4",
    consumer="blink-example",
    config={
        CH1_LINE: gpiod.LineSettings(
            direction=Direction.OUTPUT, output_value=Value.ACTIVE
        ),
        CH2_LINE: gpiod.LineSettings(
            direction=Direction.OUTPUT, output_value=Value.ACTIVE
        ),
        CH3_LINE: gpiod.LineSettings(
            direction=Direction.OUTPUT, output_value=Value.ACTIVE
        )
    },
)
    return lines

def control_gpio(lines, channel, ch_name, state):
        print (ch_name +" " + state)
        if state == "ON":
             lines.set_value(channel, Value.INACTIVE)  #not a mistake, INACTIVE indeed!
        elif state =="OFF":
             lines.set_value(channel, Value.ACTIVE)
       
        

# Initialize GPIO lines
initialized_lines = initialize_gpio()
control_gpio(initialized_lines, CH1_LINE,"Channel 1", "OFF") # just in case someone forgot to switch relay off ;)

#-----------------------------------------------------------------------------------------------
while True:
  frame=  picam2.capture_array() # Read the video frames
  frame_number +=1
  # Crop the frame using the specified coordinates
  cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
  background_frame = cropped_frame
  gray = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2GRAY)

# ARTIFACT DETECTOR
  osd_status = "ON OVERWATCH"
  print (osd_status)


# Calculate average colors of cells in the current frame
  current_avg_colors = calculate_average_intensity(gray, cell_size)

  # Calculate absolute difference between current and background average colors
  diff_avg_colors = np.abs(current_avg_colors - background_avg_colors)
  
# Threshold the difference to identify cells with significant changes
  changed_cells = (diff_avg_colors > background_threshold).any(axis=0)  # Use axis=0 instead of axis=1

  
# Check if motion is detected and display appropriate text
  if changed_cells.any():
      #OBJECT_DETECTION  
      osd_status ="OBJECT DETECTION"
      print (osd_status)
      

      
      
      results = model.track(cropped_frame, persist=True, conf=model_confidence, iou=model_iou, imgsz=[model_width,model_height], tracker=model_tracker) #show=True, tracker="botsort.yaml") 
      if results is not None:
          boxes = results[0].boxes.xywh.cpu()
          #track_ids = results[0].boxes.id.int().cpu().tolist()
          confs = results[0].boxes.conf.float().cpu().tolist()
          clss = results[0].boxes.cls.cpu().tolist()
          namess = results[0].names
      
       #if results[0].boxes.id is not None:
       #   track_ids = results[0].boxes.id.int().cpu().tolist()
       #else:
       #   track_ids = ["-" for _ in range(len(boxes))]
       #for box, track_id, conf, cls in zip(boxes, track_ids, confs, clss):
          for box, conf, cls in zip(boxes, confs, clss):
              x, y, w, h = box
              x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)

             
                
              if int(cls) == 1:  # norman
                  cv2.rectangle(cropped_frame, (x1, y1), (x2, y2), (168, 80, 50), 3)
              if int(cls) == 2:  # bob
                  cv2.rectangle(cropped_frame, (x1, y1), (x2, y2), (80, 50, 168), 3)
                      
              if int(cls) == 0:  # larry
                  cv2.rectangle(cropped_frame, (x1, y1), (x2, y2), (46, 23, 163), 3)
              #label = namess[cls] + " #" + str(track_id) + " " + str(format(conf*100, '.0f'))+"%"
              label = namess[cls] +  " " + str(format(conf*100, '.0f'))+"%"
              t_size = cv2.getTextSize(label, 0, fontScale=0.4, thickness=1)[0]
              if int(cls) == 1:  # norman
                 cv2.rectangle(cropped_frame, (x1, y1 - t_size[1] - 3), (x1 + t_size[0], y1 + 3), (168, 80, 50), -1)
              if int(cls) == 2:  # bob
                 cv2.rectangle(cropped_frame, (x1, y1 - t_size[1] - 3), (x1 + t_size[0], y1 + 3), (80, 50, 168), -1)
                 
                 
              if int(cls) == 0:  # larry
                 cv2.rectangle(cropped_frame, (x1, y1 - t_size[1] - 3), (x1 + t_size[0], y1 + 3), (46, 23, 163), -1)


              cv2.putText(cropped_frame, label, (x1, y1 - 2), 0, 0.4, [189, 189, 189], thickness=1, lineType=cv2.LINE_AA)
  else:
	  print (osd_status)
  cv2.putText(cropped_frame, osd_status, (20, 20), 0, 0.5, [46, 23, 163], thickness=1, lineType=cv2.LINE_AA)	  
  output_video.write(cropped_frame)


  cv2.imshow("Live Inference", cropped_frame)
  # Wait 1ms for ESC to be pressed
  key = cv2.waitKey(1)
  if key == 27:
     break


# Release video source and writer
output_video.release()
cv2.destroyAllWindows()
print(f"wrote file {output_filename}, frames={frame_number}")

