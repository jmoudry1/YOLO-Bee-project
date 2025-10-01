import time
import gpiod
from gpiod.line import Direction, Value

CH1_LINE = 26
CH2_LINE = 20
CH3_LINE = 21

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
             lines.set_value(channel, Value.INACTIVE)
        elif state =="OFF":
             lines.set_value(channel, Value.ACTIVE)
       
        

# Initialize GPIO lines
initialized_lines = initialize_gpio()

# Call the function to control GPIO lines
control_gpio(initialized_lines, CH1_LINE,"Channel 1", "ON")
time.sleep(5)
control_gpio(initialized_lines, CH1_LINE,"Channel 1", "OFF")
time.sleep(1)
control_gpio(initialized_lines, CH2_LINE,"Channel 2", "ON")
time.sleep(1)
control_gpio(initialized_lines, CH2_LINE,"Channel 2", "OFF")
time.sleep(1)
control_gpio(initialized_lines, CH3_LINE,"Channel 3", "ON")
time.sleep(1)
control_gpio(initialized_lines, CH3_LINE,"Channel 3", "OFF")
time.sleep(1)



# Release control over GPIO lines when done
initialized_lines.release()
