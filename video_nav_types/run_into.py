import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision
from PIL import Image
import random



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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
            nn.ReLU()
        )

        self.dense2 = nn.Sequential(
            nn.Linear(in_features=128, out_features=1),
            nn.ReLU()
        )
        self.dense3 = nn.Sequential(
            nn.Linear(in_features=128, out_features=1),
            nn.ReLU()
        )
        self.dense4 = nn.Sequential(
            nn.Linear(in_features=128, out_features=1),
            nn.ReLU()
        )
        self.dense5 = nn.Sequential(
            nn.Linear(in_features=128, out_features=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.dense_layers(x)
        a = self.dense1(x)
        b = self.dense2(x)
        c = self.dense3(x)
        d = self.dense4(x)
        e = self.dense5(x)

        return a, b, c ,d, e


model_name = "/home/nathan/Desktop/oscar_ml/run_into.pt"

image_shape = (100, 100)
def detect(img, markup = True):
    img = cv2.resize(img, image_shape, interpolation = cv2.INTER_AREA)

    color_coverted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      
    pil_image = Image.fromarray(color_coverted)

    pil_image = train_transform(pil_image)
    pil_image = pil_image.unsqueeze(0)

    res = model(pil_image)
    res = [float(i) for i in res]
    
    # ["open", "weed", "corn", "stop", "inRow",  "centerline"]:
    if res[0] < res[2] or res[0] < res[3]:
        return True
    else:
        return False
    



    return res


model = Net()

model.load_state_dict(torch.load(model_name, map_location=torch.device(device)))


model.eval()



train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(image_shape),
    transforms.Normalize(mean=[0.5], std=[0.5])
])