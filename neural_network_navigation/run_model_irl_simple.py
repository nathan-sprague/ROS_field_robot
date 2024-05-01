import cv2
from ultralytics import YOLO
import numpy as np
import torch
import torch.nn as nn
import ffmpegcv
import serial
import serial.tools.list_ports


port_name = [tuple(p) for p in list(serial.tools.list_ports.comports())][0][0]





model_name = "/home/nathan/Desktop/old_work/oscar_ml/guardrail_seg_only_april15.pt"
nav_model_name = "/home/nathan/Desktop/old_work/oscar_ml/ROS_field_robot_all_ml/simple_nav_model_reinforcement_convolutional_low_close_rand_start.pt"

class NetCV_Conv(nn.Module):
    def __init__(self):
        super(NetCV_Conv, self).__init__()
        num_classes = 2
        # Define the convolutional layers for image processing
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        in_size = 3200
        in_size = 1152

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=in_size, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1),
        )

    def forward(self, x):
        x = x.unsqueeze(0)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.linear_layers(x)
        return x


device = serial.Serial(port=port_name, baudrate=115200, timeout=.1)





# resolution = (320, 240)
# resolution=(1344b, 376) # WVGA
resolution=(2560, 720) # 720p
# resolution=(3840, 1080) # 1080p
# resolution=(4416, 1242) # 2.2K



cap = cv2.VideoCapture(2) # might need to change depending on whether you have a webcam already

cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
cap.set(cv2.CAP_PROP_FPS, 15)


writer = ffmpegcv.VideoWriter("vid_" + str(int(np.random.random()*10000)) + ".mkv", 'h264', 10)



nav_model = NetCV_Conv()
nav_model.eval()
nav_model.zero_grad()
nav_model.load_state_dict(torch.load(nav_model_name, map_location='cpu'))


model = YOLO(model_name)

def detect(img, imageSize=416):
    h, w = img.shape[0:2]
    mask = np.zeros((imageSize, imageSize, 1), np.uint8)
    results = model.predict(source=cv2.resize(img, (imageSize, imageSize), interpolation = cv2.INTER_AREA), save=False, conf=0.2)
    res = []
    for result in results:
        if result.masks is not None:
            for j, i in enumerate(result.masks.xy):
                l= [[int(i[n,0]), int(i[n,1])] for n in range(i.shape[0])]
                if len(l)>0:
                    cv2.fillPoly(mask, [np.array(l)], 255)
    cv2.imshow("mask", mask)
    return mask



while True:

    ret, img = cap.read()
    img = img[:, 0:img.shape[1]//2]
    img_og = img.copy()

    mask = detect(img)
    if np.max(mask)!=0:
        img = cv2.resize(mask, (50, 50))
        inp = torch.tensor(img/255, dtype=torch.float32)
        inp = inp.unsqueeze(0)

        with torch.set_grad_enabled(False):
            speeds = nav_model(inp)
        s1 = -float(speeds[0])
        target_speed = (1-s1/2, 1+s1/2)

    else:
        target_speed = [0,0]

    device.write(bytes("w" + str(target_speed[0]) + "," + str(target_speed[1]), 'utf-8'))

    writer.write(img_og)
    cv2.imshow("img", img_og)
    
    if cv2.waitKey(1) == 27:
        break


writer.release()
cap.release()
