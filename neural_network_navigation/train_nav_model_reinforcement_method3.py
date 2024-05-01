"""
Attempts to train with memory by adding state as nn output. 
Also simplifies everything a bit by using different coordinate system

"""


import numpy as np
import cv2
import torch
import torch.nn as nn
import video_simulator2
import time
from threading import Thread
import math
import pyproj
# import ffmpegcv

import copy


num_epochs = 20


times = {"vid_gen":[0,0], "loss":[0,0], "backProp":[0,0], "show": [0,0]}


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

guardrails = [[1,1+i*30] for i in range(4)]

guardrail_region = [[8, 2], [8, 9]]

robot_x, robot_y = 3, 3
robot_heading = math.pi*1.1


destination = [6,3]


def draw_map(cameraView, cameraLocation):
    global destination
    # print(guardrails)
    b = np.zeros((400,400,3),np.uint8)
    b[:] = (255,255,255)
    scale = 40 # pixels

    for i in range(8):
        cv2.line(b, (i*scale, 0), (i*scale, 400), (100,100,100), 1)
        cv2.line(b, (0, i*scale), (400, i*scale), (100,100,100), 1)

    
    cv2.line(b, (int(guardrail_region[0][0]*scale), b.shape[0]-int(guardrail_region[0][1]*scale)), (int((guardrail_region[1][0]-0.1)*scale), b.shape[0]-int((guardrail_region[1][1]-0.1)*scale)), (50,50,50), 4)

    robot_x, robot_y = cameraLocation[0:2]
    robot_heading = cameraView[0]

    cv2.rectangle(b, (int(robot_x*scale), b.shape[0]-int(robot_y*scale)), (int(robot_x*scale), b.shape[0]-int(robot_y*scale)), (0,255,0), 10)

    cv2.arrowedLine(b, (int(robot_x*scale), b.shape[0]-int(robot_y*scale)), (int(robot_x*scale+math.cos(-robot_heading-math.pi/2)*40), b.shape[0]-int(robot_y*scale+math.sin(-robot_heading-math.pi/2)*40)), (0,0,200), 2)


    cv2.rectangle(b, (int(destination[0]*scale), b.shape[0] - int(destination[1]*scale)), (int(destination[0]*scale), b.shape[0] - int(destination[1]*scale)), (255,0,0),6)
    cv2.imshow("map", b)


    # cv2.waitKey(0)




# model = Net()




cam = video_simulator2.Camera()


x = 0
y = 0

# guardrail_region[0][0], guardrail_region[0][1], guardrail_region[1][0], guardrail_region[1][1]
video_simulator2.makeGuardrail(cam, guardrail_region[0][0], guardrail_region[0][1], guardrail_region[1][0], guardrail_region[1][1], triangleCount=0)



cameraLocation = [robot_x, robot_y, 0.5]
cameraView = [robot_heading,0]

# cam.draw3D(cameraView, cameraLocation)
# cam.draw2D(cameraView, cameraLocation)
# draw_map()
# cv2.waitKey(0)






def run_sim(cameraLocation, cameraView, show=False, iterations=20):
    global destination
    # show = True
    losses = [{"heading": [], "speeds": [], "accomplishment": 1}]
    guardrail_hits = 0

    stage = 0

    iterations_left = 100
    while iterations_left > 0:
        iterations_left -= 1
        # print(cameraView, cameraLocation)
        img = cam.draw3D(cameraView, cameraLocation, mask=True)
        if show:
            cv2.imshow("3d", img)
            draw_map(cameraView, cameraLocation)
            cv2.waitKey(1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (50, 50))

        img = img.astype(np.float32)/255

        if np.max(img) == 0:
            print("no detections")
            losses[stage]["accomplishment"] = 2
            break

        # cv2.imshow("im", img)
        
        inp = torch.tensor(img, dtype=torch.float32)
        inp = inp.unsqueeze(0)


        speeds = model(inp)


        s1 = float(speeds[0].item())
        s2 = 0.02#float(speeds[1]) / 2

        cameraView[0] += s1*0.1
        # print("s1", s1)
        cameraLocation[0] += math.cos(-cameraView[0]-math.pi/2) * s2
        cameraLocation[1] += math.sin(-cameraView[0]-math.pi/2) * s2


        if stage == 0: # line up approach
            destination = [6,3]

            
            if cameraLocation[0] > 6:
                # print("reached destination 0")

                if 2 < cameraLocation[1] < 3:
                    losses[stage]["accomplishment"] = 0.7

                stage = 1

                iterations_left += 500

                losses.append({"heading": [], "speeds": [], "accomplishment": 1})

        elif stage == 1:


            if cameraLocation[0] > 7:
                # print("reached destination 1")

                if 2 < cameraLocation[1] < 3:
                    losses[stage]["accomplishment"] = 0.9

                losses.append({"heading": [], "speeds": [], "accomplishment": 1})

                iterations_left += 500
                stage = 2

            destination = [7.5, 3]
        else:

            if cameraLocation[1] > 9:
                # print("reached destination 3")
                if 7.5 < cameraLocation[0] < 8:
                    losses[stage]["accomplishment"] = max(min(1000/-iterations_left, 0.5), 0.8)
                break


            destination = [7.5, cameraLocation[1]+1]

        if 2 < cameraLocation[1] < 9 and 7.7 < cameraLocation[0] < 8.2:
            # print("hit guardrail", )
            if math.degrees(cameraView[0]) > 270:
                cameraView[0] += 0.1
            else:
                cameraView[0] -= 0.1

            guardrail_hits += 1
            cameraLocation[0] = 7.8

        ideal_heading = (-math.atan2(cameraLocation[1]-destination[1], cameraLocation[0]-destination[0])+math.pi/2) % (2*math.pi)

        cameraView[0] %= (2*math.pi)
        angle_off = cameraView[0] - ideal_heading
        if angle_off > math.pi:
            angle_off = 2*math.pi - (cameraView[0] - ideal_heading)
        elif angle_off < -math.pi:
            angle_off += 2*math.pi

        losses[stage]["heading"].append(abs(angle_off))

        
        speed_error = abs(-angle_off/2 - speeds[0])
        if (angle_off > 0 and s1 > 0) or (angle_off < 0 and s1 < 0): # same sign
            if speed_error > 0.2:
                speed_error = speed_error/speed_error * 0.2
            # speed_error = min(speed_error, 0.2)


        losses[stage]["speeds"].append(speed_error)

    
    loss = 0
    for stage, l in enumerate(losses):

        speed_loss = sum(l["speeds"]) / len(l["speeds"])  * 5
        heading_loss = sum(l["heading"]) / len(l["heading"]) / (3-stage)
        accomplishment_loss = l["accomplishment"]
        if show:
            print("stage", stage, "mean speed loss", speed_loss, "heading", heading_loss, "accomplish", accomplishment_loss)

        loss += speed_loss * accomplishment_loss
        # loss += heading_loss
        # loss += accomplishment_loss
    loss -= len(losses)*0.5

    if guardrail_hits > 0:
        loss += 2

    # loss = torch.tensor(loss, requires_grad=True)



    # loss = sum(losses) / len(losses) + speeds[0]*0


    return loss




model = NetCV_Conv()
# model.load_state_dict(torch.load("/home/nathan/Desktop/old_work/oscar_ml/ROS_field_robot_all_ml/simple_nav_model_reinforcement_ideal_convolutional.pt", map_location='cpu'))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


all_losses = []
show = False
for epoch in range(num_epochs):
    
    reached_count = 0
    dists_reached=[]
    losses = []
    for i in range(5):
        cameraLocation = [3+np.random.random()*2, 3+np.random.random()*2, 0.5]
        cameraView = [math.pi*1.5,0]
        model.train()
        l = run_sim(cameraLocation, cameraView, show=(i==4), iterations=100)
        losses.append(l)

    # times["backProp"][1] = time.time()
    

    loss = sum(losses)/len(losses)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # times["backProp"][0] = time.time()-times["backProp"][1]


    print("epoch", epoch+1, "loss", float(loss))

    # if len(all_losses) > 0 and float(loss) < min(all_losses):
    #     best_model_wts = copy.deepcopy(model.state_dict())
    # elif len(all_losses) > 10 and min(all_losses[-5::]) > min(all_losses)*1.2:
    #     print("Reverting back")

    #     model.load_state_dict(best_model_wts)


    all_losses.append(loss.item())
    if len(all_losses) > 0:
        min_l, max_l = min(all_losses)*0.8, max(all_losses)*1.2
        b = np.zeros((400,400,3), np.uint8)
        b[:] = (255,255,255)
        lastLs = [0,0]
        for i, ls in enumerate(all_losses):
            if i>0:
                cv2.line(b, (int(400*(i+1)/epoch), int(400-400*(ls-min_l)/(max_l-min_l))), (int(400*i/epoch), int(400-400*(lastLs-min_l)/(max_l-min_l))), (0,0,255),2)
            lastLs = ls
        cv2.imshow("pl", b)
        k = cv2.waitKey(100)
        if k == 32:
            show = not show
        elif k == 13:
            print("done. Saving")
            break


        # total_time = times["vid_gen"][0] + times["loss"][0] + times["backProp"][0] + times["show"][0]
        # print("total time", total_time)
        # print("total", total_time, "vid gen", int(times["vid_gen"][0]/total_time*100), "loss", int(times["loss"][0]/total_time*100), 
        #      "back prop", int(times["backProp"][0]/total_time*100), "show", int(times["show"][0]/total_time*100))
    times["backProp"][0] += time.time() - times["backProp"][1]


            

# color_writer.release()

    
torch.save(model.state_dict(), "simple_nav_model_reinforcement_convolutional_low_close_rand_start.pt")