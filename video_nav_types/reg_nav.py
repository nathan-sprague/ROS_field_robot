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


vid_name = "/home/nathan/Desktop/oscar_ml/dataset/july6_6.mkv"
image_shape = (100, 100)
model_name = "reg_july13_3.pt"




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
    transforms.Resize(image_shape),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def detect(img, markup=True):

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



model = Net().to(device)

model.load_state_dict(torch.load(model_name, map_location=torch.device(device)))


model.eval()


if __name__ == "__main__":
    class1_count = 0

    class2_count = 0
    cap = cv2.VideoCapture(vid_name)
    while True:
        # img = cv2.imread("/home/nathans/Desktop/random/images_open1688797498/" + imgName)
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        if not ret:
            break
        res = detect(img)
        cv2.imshow("img", img)
        if cv2.waitKey(0) == 27:
            cap.release()
            break  
