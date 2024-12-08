import cv2
import numpy as np
import threading
from tkinter import Tk, Button, Label, Canvas, Frame, BOTH
from PIL import Image, ImageTk
from pycoral.adapters import common, detect
from pycoral.utils.edgetpu import make_interpreter
from adafruit_servokit import ServoKit

# Load model
model_path = 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite'
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize servos
kit = ServoKit(channels=16)
servo_x = kit.servo[0]  # Horizontal axis
servo_y = kit.servo[1]  # Vertical axis
servo_x.angle = 90
servo_y.angle = 90

# Global variables for object tracking
tracked_objects = []
current_target_index = 0
x_center = 90
y_center = 90

# Function to move servos
def move_servos(x, y):
    global x_center, y_center
    servo_x.angle = int(np.clip(x, 0, 180))
    servo_y.angle = int(np.clip(y, 0, 180))
    x_center = x
    y_center = y

# Function to select the next target
def select_next_target():
    global current_target_index
    if tracked_objects:
        current_target_index = (current_target_index + 1) % len(tracked_objects)
        update_target_display()

# Function to select the previous target
def select_previous_target():
    global current_target_index
    if tracked_objects:
        current_target_index = (current_target_index - 1) % len(tracked_objects)
        update_target_display()

# Function to update the target display
def update_target_display():
    if tracked_objects:
        obj = tracked_objects[current_target_index]
        bbox = obj.bbox.scale(frame.shape[1] / resized_frame.shape[1], frame.shape[0] / resized_frame.shape[0])
        x_center_obj = int((bbox.xmin + bbox.xmax) / 2)
        y_center_obj = int((bbox.ymin + bbox.ymax) / 2)
        move_servos(x_center_obj, y_center_obj)
        target_label.config(text=f"Tracking Object {current_target_index + 1}/{len(tracked_objects)}")

# Tkinter UI setup
root = Tk()
root.attributes('-fullscreen', True)  # Fullscreen mode
root.bind("<Escape>", lambda e: root.destroy())  # Exit fullscreen with ESC key

# Frame for video display
frame = Frame(root, bd=2, relief="sunken")
frame.pack(fill=BOTH, expand=1)
canvas = Canvas(frame, bg="black")
canvas.pack(fill=BOTH, expand=1)

prev_button = Button(root, text="Previous Target", command=select_previous_target)
prev_button.pack(side="left")

next_button = Button(root, text="Next Target", command=select_next_target)
next_button.pack(side="right")

target_label = Label(root, text="No targets detected")
target_label.pack()

# Function to update the video feed on the canvas
def update_video_feed():
    global frame
    ret, frame = cap.read()
    if ret:
        # Resize frame for model input
        resized_frame = cv2.resize(frame, common.input_size(interpreter))
        common.set_input(interpreter, resized_frame)

        # Run inference
        interpreter.invoke()
        objs = detect.get_objects(interpreter, score_threshold=0.5)
        tracked_objects = objs

        # Draw bounding boxes and update target
        for obj in objs:
            bbox = obj.bbox.scale(frame.shape[1] / resized_frame.shape[1], frame.shape[0] / resized_frame.shape[0])
            cv2.rectangle(frame, (int(bbox.xmin), int(bbox.ymin)), (int(bbox.xmax), int(bbox.ymax)), (0, 255, 0), 2)

        if tracked_objects:
            update_target_display()

        # Convert frame to PIL Image and display it on the canvas
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img)
        canvas.create_image(0, 0, anchor="nw", image=img_tk)
        canvas.img_tk = img_tk  # Keep reference to avoid garbage collection

    root.after(10, update_video_feed)  # Schedule the function to run again

# Run the video feed update in the main thread
update_video_feed()

# Start the Tkinter main loop
root.mainloop()

# Release video capture
cap.release()
cv2.destroyAllWindows()
