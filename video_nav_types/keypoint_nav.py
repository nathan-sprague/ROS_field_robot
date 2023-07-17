import cv2
from ultralytics import YOLO
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import os
import cv2
import numpy as np
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision
from PIL import Image
import ffmpegcv


modelName = "ground_kp_july5_2.pt"
detector = YOLO(modelName)
image_shape = (100, 100)
model_name = "reg_july13_3.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        num_classes = 2
        # Define the convolutional layers for image processing
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
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

        self.dense_layers = nn.Sequential(
            nn.Linear(in_features=4608, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            # nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
        )

        self.dense1 = nn.Sequential(
            nn.Linear(in_features=128, out_features=1),
            # nn.Sigmoid()
            nn.ReLU()
        )

        # self.dense2 = nn.Sequential(
        #     nn.Linear(in_features=128, out_features=1),
        #     # nn.Sigmoid()
        #     nn.ReLU()
        # )
        # self.dense3 = nn.Sequential(
        #     nn.Linear(in_features=128, out_features=1),
        #     # nn.Sigmoid()
        #     nn.ReLU()
        # )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.dense_layers(x)
        a = self.dense1(x)
        # b = self.dense2(x)
        # c = self.dense3(x)
 

        return a#, b, c


def detect(img, markup=True):

		
		results = detector.predict(source=cv2.resize(img, (416, 416), interpolation = cv2.INTER_AREA), show=False, save=False, conf=0.05)

		earList = []

		imageSize = 416

		inside_row = False
		robot_pose = 0
		possible_rows = []
		about_to_hit = False

		h, w = img.shape[0:2]
		for result in results:
			
			for det in range(len(result.boxes.xyxy)):
				
				i, c = result.boxes.xyxy[det], result.boxes.conf[det]
				color = (255,0,0)
				x1, y1, x2, y2 = max(min(int(i[0]), int(i[2])),0), max(min(int(i[1]), int(i[3])), 0), min(max(int(i[0]), int(i[2])), 480), min(max(int(i[1]), int(i[3])), 480)
				x1 = int(x1 * w / imageSize)
				y1 = int(y1 * h / imageSize)
				x2 = int(x2 * w / imageSize)
				y2 = int(y2 * h / imageSize)

				if x1 == 10 or x2 > w-10:
					continue




				kp = result.keypoints.xy[det]

				pt0 = (int(float(kp[0][0])*w/imageSize), int(float(kp[0][1])*h/imageSize))
				pt1 = (int(float(kp[1][0])*w/imageSize), int(float(kp[1][1])*h/imageSize))
				pt2 = (int(float(kp[2][0])*w/imageSize), int(float(kp[2][1])*h/imageSize))

				if 2*w/3 < pt1[0] or pt1[0] < w/3 or pt2[1] < h*0.8:
					continue

				if markup:
					print("drawing")
					cv2.line(img, (img.shape[1]//2, 0), (img.shape[1]//2, img.shape[0]), (0,0,0), 2)
					cv2.line(img, (pt0[0], pt0[1]), (pt1[0], pt1[1]), (255,0,0), 2)
					cv2.line(img, (pt2[0], pt2[1]), (pt1[0], pt1[1]), (255,0,0), 2)

					cv2.rectangle(img, (pt0[0], pt0[1]), (pt0[0], pt0[1]), (0,255,0), 5)
					cv2.rectangle(img, (pt1[0], pt1[1]), (pt1[0], pt1[1]), (0,255,255), 5)
					cv2.rectangle(img, (pt2[0], pt2[1]), (pt2[0], pt2[1]), (0,255,0), 5)



					center = (int((x1+x2)/2), int((y1+y2)/2))
					robot_pose = (pt1[0]/w - 0.5) * 90
					inside_row = True

					# ll = int(min(abs(x2-x1), abs(y2-y1)))
					# cv2.arrowedLine(img, center, (int(center[0]+ll*math.cos(math.radians(angle-90))), int(center[1]+ll*math.sin(math.radians(angle-90)))), (0,255,255), 2)
					
		if not inside_row:

			return detect_alt(img)
			
		# self.inside_row, self.robot_pose, self.possible_rows, self.about_to_hit = self.nav_tool.detect(img, markup=show_stream)
		print("robot pose", robot_pose)
		return inside_row, robot_pose, possible_rows, about_to_hit

def detect_alt(img, markup=True):

    img_resize = cv2.resize(img, image_shape, interpolation = cv2.INTER_AREA)
    img_resize2 = cv2.flip(img_resize, 1)

    color_coverted = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB) 
    pil_image = Image.fromarray(color_coverted)
    pil_image = train_transform(pil_image)
    # pil_image = pil_image.unsqueeze(0).to(device)

    color_coverted2 = cv2.cvtColor(img_resize2, cv2.COLOR_BGR2RGB) 
    pil_image2 = Image.fromarray(color_coverted2)
    pil_image2 = train_transform(pil_image2)
    # pil_image2 = pil_image2.unsqueeze(0).to(device)

    res = model(torch.stack([pil_image, pil_image2]))

    if markup:

        cv2.line(img, (img.shape[1]//2, 0), (img.shape[1]//2, img.shape[0]), (0,0,0), 2)

        cv2.line(img, (int(res[0]*img.shape[1]), 0), (int(res[0]*img.shape[1]), img.shape[0]), (0,255,0), 2)

        cv2.line(img, (int((1-res[1])*img.shape[1]), 0), (int((1-res[1])*img.shape[1]), img.shape[0]), (0,0,255), 2)

        res = [(res[0]+(1-res[1]))/2]

        cv2.line(img, (int(res[0]*img.shape[1]), 0), (int(res[0]*img.shape[1]), img.shape[0]), (255,0,0), 2)

    # torch.Size([32, 3, 100, 100])
    # return res
    inside_row = True
    possible_rows = []
    about_to_hit = False
    robot_pose = (res[0] - 0.5) * 90

    return inside_row, robot_pose, possible_rows, about_to_hit




train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(image_shape),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


model = Net().to(device)

model.load_state_dict(torch.load(model_name, map_location=torch.device(device)))


model.eval()


if __name__ == "__main__":
	vidName = "/home/nathan/Desktop/oscar_ml/random_driving_dataset/random_july7_8.mkv" # green
	# vidName = "/home/nathan/Desktop/ear_stalk_detection/datasets/ear/ear_rgb_training_videos/color (copy 21) (copy 1).avi" # not green
	vidName = "/home/nathan/Desktop/oscar_ml/dataset/realsense2022_2.mkv"
	# colorWriter = ffmpegcv.VideoWriter("kp.mp4", "h264", 10)

	cap = cv2.VideoCapture(vidName)
	# cap = cv2.VideoCapture(0)
	framenum = 2750
	cap.set(1, framenum)

	while True:
		ret, img = cap.read()
		# img = img[:, 0:img.shape[1]//2]
		if not ret:
			break
		detect(img, markup=True)
		img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
		cv2.imshow("img", img)
		# colorWriter.write(img)
		print(framenum)
		framenum += 1
		cap.set(1, framenum)
		if cv2.waitKey(100) == 27:
			break
	cap.release()
	# colorWriter.release()