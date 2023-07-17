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
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


vid_name = "file:///home/nathans/Desktop/oscar_ml/dataset/july5_24.mkv"
image_shape = (100, 100)
model_name = "/home/nathans/Desktop/random/reg_july13.pt"




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





train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(image_shape, antialias=None),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def run_model(img):
    img = cv2.resize(img, image_shape, interpolation = cv2.INTER_AREA)

    color_coverted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      
    pil_image = Image.fromarray(color_coverted)

    pil_image = train_transform(pil_image)
    pil_image = pil_image.unsqueeze(0).to(device)
    print(pil_image.shape)
    res = model(pil_image)
    # torch.Size([32, 3, 100, 100])
    return res


model = Net().to(device)

model.load_state_dict(torch.load(model_name, map_location=torch.device(device)))


model.eval()


class1_count = 0

class2_count = 0
cap = cv2.VideoCapture(vid_name)
while True:
    # img = cv2.imread("/home/nathans/Desktop/random/images_open1688797498/" + imgName)
    ret, img = cap.read()
    if not ret:
        break
    res = run_model(img)
    res = [float(i) for i in res]
    print(res)
    cv2.line(img, (img.shape[1]//2, 0), (img.shape[1]//2, img.shape[0]), (0,0,0), 2)
    cv2.line(img, (int(res[0]*img.shape[1]), 0), (int(res[0]*img.shape[1]), img.shape[0]), (255,0,0), 2)
    # print("prediction: open:", res[0], "weed", res[1], "corn", res[2])
    # cv2.line(img, (int(res[0]*img.shape[1]), int(res[2]*img.shape[0])), (int(res[1]*img.shape[1]), img.shape[0]), (255,0,0), 2)
    img = cv2.resize(img, (300, 200))
    cv2.imshow("img", img)
    k = cv2.waitKey(0)
    if k == 27:
        break
