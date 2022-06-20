from numpy.core.defchararray import center
import pyrealsense2 as rs
import statistics

import numpy as np
import cv2
import argparse  # Import argparse for command-line options
import os.path
import math
import time
import random

_stepsShown = []  # [1,2,3,4,5,6]

# _filename = "Desktop/real_corn_july_15/human.bag"
# _filename = "tall/rs_1629482768.bag"
_filename = "/Users/nathan/bag_files/rs_1629481328.bag"
# _filename = "Desktop/fake_track/object_detection6.bag"
# _filename=""
_useCamera = False
_showStream = True
_saveVideo = False
_realtime = True
_startFrame = 300
_rgbFilename = "blah.mp4"
_navTypes = ["eStop"]


def morph_shape(val):
    if val == 0:
        return cv2.MORPH_RECT
    elif val == 1:
        return cv2.MORPH_CROSS
    elif val == 2:
        return cv2.MORPH_ELLIPSE


class RSCamera:
    def __init__(self, useCamera, saveVideo, filename="", rgbFilename="", realtime=True, startFrame=0, stepsShown=[]):
        self.useCamera = useCamera
        self.saveVideo = saveVideo
        self.realtime = realtime
        self.startFrame = startFrame
        self.stop = False
        self.heading = 0
        self.smoothCenter = -1000
        self.flagCenter = -1000
        self.legDist = 1000
        self.stepsShown = stepsShown
        self.lastLeft1 = []
        self.lastLeftColors = []
        self.lastLeft2 = []
        self.lastRight1 = []
        self.lastRight2 = []
        self.stalkCoords = []

        # this is needed for the floor detection function. I don't want to make the same image each time so I do it here. 
        # comment it out if you dont use the floor detection and need maximum memory
        h = 480
        w = 640

        #  rampr8 = np.linspace(1000/256, 70000/256, h)
        rampr8 = np.linspace(8000, 0, int(h * 0.6))
        rampr8 = np.tile(rampr8, (w, 1))
        rampr8 = cv2.rotate(cv2.merge([rampr8]), cv2.ROTATE_90_CLOCKWISE)

        rampr16 = (rampr8).astype('uint16')
        rampr8 = rampr8.astype('uint8')
        b = np.zeros((h, w), np.uint8)
        bb = np.zeros((h, w), np.uint16)
        hh = rampr8.shape[0]
        ww = rampr8.shape[1]

        b[h - hh:h, 0:ww] = rampr8
        rampr8 = b
        r = h - 1
        m = 0
        values = [0] * 190
        i = 0
        while i < len(values):
            values[i] = int(230.0 / 190.0 * i)

            #   print(values[i])
            i += 1

        while r > 0:
            l = rampr8[r, 0]
            r -= 1
        print(m)

        bb[h - hh:h, 0:ww] = rampr16
        self.rampr16 = bb
        self.ramprJet = cv2.applyColorMap(rampr8, cv2.COLORMAP_JET)
        #  print(self.ramprJet[0:1, 0:w])
        # cv2.imshow("j", self.ramprJet)
        # cv2.imshow("r", self.rampr16)
        self.rgbFilename = rgbFilename

        if useCamera:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            if saveVideo:
                if filename == "":
                    filename = 'object_detections/rs_' + str(int(time.time())) + '.bag'
                print("saving video as " + filename)
                self.config.enable_record_to_file(filename)
            else:
                print("WARNING: Using camera but NOT saving video. Are you sure?\n\n\n\n\n\n")

        elif True:
            # Create object for parsing command-line options
            parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                          Remember to change the stream fps and format to match the recorded.")
            # Add argument which takes path to a bag file as an input
            parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
            # Parse the command line arguments to an object
            args = parser.parse_args()
            args.input = filename
            # Safety if no parameter have been given
            if not args.input:
                print("No input parameter have been given.")
                print("For help type --help")
                exit()
            # Check if the given file have bag extension
            if os.path.splitext(args.input)[1] != ".bag":
                print("The given file is not of correct file format.")
                print("Only .bag files are accepted")
                exit()
            try:
                # Create pipeline
                self.pipeline = rs.pipeline()

                # Create a config object
                self.config = rs.config()

                # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
                rs.config.enable_device_from_file(self.config, args.input)

                # Configure the pipeline to stream the depth stream
                # Change this parameters according to the recorded bag file resolution

                self.config.enable_stream(rs.stream.depth, rs.format.z16, 30)
                self.config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

            finally:
                pass

    def floorDetection(self, depth_image, depth_color_image, color_image, showStream):
        w = depth_image.shape[1]
        h = depth_image.shape[0]

        depth_image[depth_image > 65530] = 0

        l = 3
        stalkLocationsLeft = []
        stalkLocationsRight = []
        left1 = []
        left2 = []
        right1 = []
        right2 = []
        consecutiveCenters = 0

        potentialCenters = []

        nonChanges = 0
        xx = 0
        ijk = 0

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        centers = []
        xx = int(w * 0.25)
        l = 3
        while xx < w * 0.75:
            if xx < w / 2:
                hh = h - (h * 0.5) * xx / (w / 2)
            else:
                hh = h / 2 + (h * 0.5) * (xx - w / 2) / (w / 2)
            hh = int(hh)
            color_b = hsv[int(h / 3):hh, xx:xx + l]
            blue = color_b.copy()
            blue[:, :, 0] = 0
            blue[:, :, 2] = 0
            centers += [np.std(blue)]
            #  cv2.line(color_image, (xx, 0), (xx, 300), (np.std(blue),np.std(blue),np.std(blue)), l)
            xx += l
        middle = centers.index(min(centers)) * l + int(w * 0.25)
        cv2.line(color_image, (middle, 0), (middle, h), (0, 0, 0), 1)
        xx = 0
        l = 3

        while xx < w:

            if xx < middle:
                hh = h - (h * 0.5) * xx / (middle)
            else:
                hh = h / 2 + (h * 0.5) * (xx - middle) / (middle)
            hh = int(hh)
            #    cv2.line(color_image, (xx, 0), (xx, int(hh)), (0,255,0), 5)

            b = depth_image[0:hh, xx:xx + l]
            # cv2.line(color_image, (xx, int(h/3)), (xx, int(hh)), (0,255,0), 5)
            color_b = gray[int(h / 3):hh, xx:xx + l]
            blue = color_b.copy()
            # blue[:, :, 0] = 0
            # blue[:, :, 2] = 0
            mm = np.std(blue)
            #    cv2.line(color_image, (xx, 0), (xx, 30), (mm*10,mm*10,mm*10), l) # standard dev

            if mm < 30:

                if xx < middle:
                    left1 += [xx]
                else:
                    right1 += [xx]
                consecutiveCenters += 1
            else:
                if consecutiveCenters > 1:
                    if xx < middle:
                        s = 0
                        cc = consecutiveCenters
                        #         print(cc, left1)
                        while cc > 0:
                            s += left1[-1]
                            left1.remove(left1[-1])
                            cc -= 1

                        left1 += [int(s / consecutiveCenters)]

                #     print("now", consecutiveCenters, left1)
                if len(left1) > 0:
                    cv2.line(color_image, (left1[-1], 0), (left1[-1], 30), (0, 0, 0), l)
                consecutiveCenters = 0
            a = b.size
            b = b[b != 0]
            c = 255 * b.size / a

            m = np.median(b[b != 0])
            mm = np.std(b[b != 0]) / 10

            cv2.line(depth_color_image, (xx, 0), (xx, 30), (mm, mm, mm), l)  # standard  dev
            cv2.line(depth_color_image, (xx, 30), (xx, 60), (c, c, c), l)  # valid elements
            #   cv2.line(depth_color_image, (xx, 60), (xx, 100), (int(n[0]), int(n[1]), int(n[2])), l) # median

            if not math.isnan(m):
                indexTo = h - int(m / 5 / 256 * 190)

                if abs(indexTo - hh) < 100 and indexTo > h * 0.5 and 0 < mm < 100:
                    certainty = int(100 / mm)
                    cv2.line(depth_color_image, (xx, indexTo), (xx, indexTo + certainty), (0, 255, 0), l)  # median

                    if xx < middle:
                        if len(stalkLocationsLeft) > 0:
                            if stalkLocationsLeft[-1][0] - indexTo > 6 and nonChanges > 0:
                                cv2.line(depth_color_image, (xx, stalkLocationsLeft[-1][0]), (xx, indexTo), (0, 0, 255),
                                         l)
                                cv2.line(color_image, (xx, 0), (xx, 30), (0, 0, 255), l)
                                left2 += [xx]
                                nonChanges = 0
                            else:
                                nonChanges += 1

                        stalkLocationsLeft += [(indexTo, xx)]
                    else:
                        if len(stalkLocationsRight) > 0:
                            if indexTo - stalkLocationsRight[-1][0] > 6 and nonChanges > 10:
                                cv2.line(depth_color_image, (xx, stalkLocationsRight[-1][0]), (xx, indexTo),
                                         (0, 0, 255), l)
                                cv2.line(color_image, (xx, 0), (xx, 30), (0, 0, 255), l)
                                right2 += [xx]
                                nonChanges = 0
                            else:
                                nonChanges += 1
                        stalkLocationsRight += [(indexTo, xx)]
                    cv2.line(depth_color_image, (xx, 60), (xx, 200), (certainty, certainty, certainty), l)

                elif mm != 0:
                    cv2.line(depth_color_image, (xx, indexTo), (xx, indexTo), (0, 0, 255), l)  # median

            xx += l
            ijk += 1
        if len(stalkLocationsLeft) == 0:
            return

        self.findStalkMovement(middle, left1, self.lastLeft1, color_image, depth_image)

        # cv2.line(color_image, (middle, 55), (int(middle + (leftSum/numSum)*10), int(55-leftSum/2)), (0, 255, 0), 2)
        #   print(self.lastLeft1, left1)

        #  cv2.imshow('6', cv2.applyColorMap(depth_image, cv2.COLORMAP_JET))

        cv2.imshow('4', color_image)
        cv2.imshow('5', depth_color_image)

        return

    def findStalkMovement(self, middle, array, oldArray, color_image, depth_image):
        ht = depth_image.shape[0]
        wt = depth_image.shape[1]
        stalkMap = np.zeros((800, 200), np.uint8)
        stalkMap = cv2.cvtColor(stalkMap, cv2.COLOR_GRAY2BGR)
        leftSum = 0
        numSum = 1
        colorLocations = []
        seenCoords = []

        for i in array:
            change = 10000
            closest = 10000
            closestLocation = [0, 0]

            for j in oldArray:

                if abs(i - j[0]) < closest:
                    closest = abs(i - j[0])
                    change = i - j[0]
                    closestLocation = j

            # i = closestLocation[0]
            j = closestLocation[0]
            #    print(i,j)
            if abs(i - j) < 20:
                leftSum += i - j
                numSum += 1
                foundColor = closestLocation[1]

            else:
                foundColor = random.randint(0, 16777215)

            n = 0
            while n < len(colorLocations):
                if colorLocations[n][1] == foundColor:
                    if closest < colorLocations[n][2]:
                        colorLocations[n][1] = random.randint(0, 16777215)
                        colorLocations[n][3] = 10000
                    else:
                        foundColor = random.randint(0, 16777215)
                        closest = 100000
                        change = 10000
                n += 1

            colorLocations += [[i, foundColor, closest, change]]

            #  b = depth_image[0:i-3, i:i+3]
            #  m = np.median(b[b!=0])

            if True:
                #  indexTo = ht - int(m/5/256*190)
                hh = int(ht - (ht * 0.5) * i / (wt / 2))
                b = depth_image[0:hh, i:i + 1]
                m = np.median(b[b != 0])
                if math.isnan(m):
                    indexTo = int(ht / 2 + ht / 2 * (middle - i) / (wt * 0.6))
                else:
                    indexTo = int(ht - int(m / 5 / 256 * 190))

                interm = 100 * (1 - 2.72 ** -((indexTo - ht / 2) / 100))
                coordY = indexTo  # int(ht-interm*4)

                if (indexTo - ht / 2) == 0:
                    indexTo = int(ht / 2 - 1)
                coordX = int((middle - i) / (indexTo - ht / 2))

                #     print(coordX, coordY, indexTo)
                cv2.line(stalkMap, (100 - coordX, 500 - coordY), (100 - coordX, 500 - coordY),
                         (foundColor % 256, (foundColor / 256) % 256, (foundColor / 65536) % 256), 5)

                cv2.line(stalkMap, (100 - coordX, 500 - coordY), (100 - coordX, 500 - coordY),
                         (foundColor % 256, (foundColor / 256) % 256, (foundColor / 65536) % 256), 5)
                #     print(i, indexTo)
                cv2.line(color_image, (i, indexTo), (i + 10, indexTo),
                         (foundColor % 256, (foundColor / 256) % 256, (foundColor / 65536) % 256), 2)
            cv2.line(color_image, (i, 30), (i, 100),
                     (foundColor % 256, (foundColor / 256) % 256, (foundColor / 65536) % 256), 2)

        movement = 0
        numMoves = 0
        for i in colorLocations:
            if abs(i[3]) < 10:
                if i[3] < 0:
                    cv2.line(color_image, (i[0], 100), (i[0] - i[3], 110), (0, 255, 0), 2)
                elif i[3] > 0:
                    cv2.line(color_image, (i[0], 100), (i[0] - i[3], 110), (0, 0, 255), 2)
                else:
                    cv2.line(color_image, (i[0], 100), (i[0], 110), (150, 150, 150), 2)
                numMoves += 1
                movement += i[3]
            else:
                cv2.line(color_image, (i[0], 100), (i[0], 110), (150, 150, 150), 2)
        if numMoves > 0:
            aveMovement = int(movement / numMoves) * 10
            cv2.line(color_image, (middle, 130), (middle + aveMovement, 100), (255, 0, 0), 2)
        self.lastLeft1 = colorLocations

        #   cv2.line(color_image, (int(middle-wt/2), int(ht)), (middle, int(ht/2)), (0,0,0), 2)

        cv2.imshow("gps", stalkMap)

    def birdsEye(self, depth_image, depth_color_image, color_image, showStream):
        ht = depth_image.shape[0]
        wt = depth_image.shape[1]

        resized = np.zeros((ht, wt), np.uint16)
        resized[:, :] = (16777215)
        #  cv2.imshow("rsz", resized)
        dim = (wt, int(ht * 0.5))
        avgFloor = np.median(depth_image[int(ht * 0.5)::, int(wt / 2) - 10:int(wt / 2) + 10], axis=1)
        avgFloor = cv2.blur(avgFloor, (50, 50))
        avgFloor = cv2.resize(avgFloor, dim, interpolation=cv2.INTER_AREA) - 100
        resized[int(ht * 0.5)::, :] = avgFloor
        #  resized[:, int(ht*0.5):ht] = cv2.resize(depth_image[int(wt/2):0, :], dim, interpolation = cv2.INTER_AREA)
        color_image[abs(cv2.subtract(resized, depth_image)) < 100] = (0, 0, 0)
        depth_image[abs(cv2.subtract(resized, depth_image)) < 100] = 0

        middle = int(wt / 2)
        stalkMap = np.zeros((500, 200), np.uint8)
        stalkMap = cv2.cvtColor(stalkMap, cv2.COLOR_GRAY2BGR)

        if True:
            x = 0
            chunkSize = 50  # ht-1
            while x < wt - wt / chunkSize:
                y = 0
                b = depth_image[:, x:x + int(wt / chunkSize)]
                m = np.median(b[b != 0])

                if not math.isnan(m) and middle != x:
                    m2 = np.max(b[b != 0])
                    m3 = np.mean(b[b != 0])
                    xSpot = 200 - (int((middle - x) / 4) + 90)
                    ySpot = 200 - int(m / 50)
                    xSpot = int((middle - x) / (m) * 500) + 100
                    ySpot2 = 200 - int(m2 / 50)
                    ySpot3 = 200 - int(m2 / 50)
                    #   print(xSpot, ySpot)
                    foundColor = random.randint(0, 16777215)
                    cv2.line(stalkMap, (xSpot, ySpot), (xSpot, ySpot),
                             (foundColor % 256, (foundColor / 256) % 256, (foundColor / 65536) % 256), 5)
                #  cv2.line(stalkMap, (xSpot, ySpot2), (xSpot, ySpot2), (foundColor%256, (foundColor/256)%256, (foundColor/65536)%256), 5)
                #  cv2.line(stalkMap, (xSpot, ySpot3), (xSpot, ySpot3), (foundColor%256, (foundColor/256)%256, (foundColor/65536)%256), 5)

                while False and y < ht:
                    b = depth_image[y:y + int(ht / chunkSize), x:x + int(wt / chunkSize)]
                    m = np.median(b[b != 0])
                    #   m=1000
                    if not math.isnan(m) and middle != x:
                        #   print("m", m)

                        #      angle = math.atan((ht-y)/(middle-x))

                        angle = (middle - x)
                        # print("angle", angle)

                        foundColor = random.randint(0, 16777215)
                        #         cv2.rectangle(color_image, (x, y), (x+int(wt/chunkSize), y+int(ht/chunkSize)), (foundColor%256, (foundColor/256)%256, (foundColor/65536)%256), 10)
                        cv2.rectangle(color_image, (x + 5, y + 5),
                                      (x - 5 + int(wt / chunkSize), y - 5 + int(ht / chunkSize)),
                                      (m / 10, m / 10, m / 10), 10)

                        #    xSpot = int(math.sin(angle) * m/50) + 100
                        #   ySpot = 200-int(math.cos(angle) * m/50)

                        xSpot = int((middle - x) / 4) + 90
                        ySpot = 200 - int(m / 50)
                        #   print(xSpot, ySpot)

                        cv2.line(stalkMap, (xSpot, ySpot), (xSpot, ySpot),
                                 (foundColor % 256, (foundColor / 256) % 256, (foundColor / 65536) % 256), 5)
                    #   cv2.line(stalkMap, (xSpot, ySpot), (xSpot, ySpot), (), 5)

                    y += int(ht / chunkSize)

                x += int(wt / chunkSize)

        #     print(coordX, coordY, indexTo)
        #     cv2.line(stalkMap, (100-coordX, 500-coordY), (100-coordX, 500-coordY), (foundColor%256, (foundColor/256)%256, (foundColor/65536)%256), 5)

        #     cv2.line(stalkMap, (100-coordX, 500-coordY), (100-coordX, 500-coordY), (foundColor%256, (foundColor/256)%256, (foundColor/65536)%256), 5)
        # #     print(i, indexTo)
        #     cv2.line(color_image, (i, indexTo), (i+10, indexTo), (foundColor%256, (foundColor/256)%256, (foundColor/65536)%256), 2)

        cv2.imshow("gps", stalkMap)
        cv2.imshow("di", depth_image)
        #   cv2.imshow("di16", resized)

        cv2.imshow("cm", color_image)

    def positionEstimation(self, depth_image, depth_color_image, color_image, showStream):
        #    self.floorDetection(depth_image, depth_color_image, color_image, showStream)
        self.birdsEye(depth_image, depth_color_image, color_image, showStream)
        return

    def apply_brightness_contrast(self, input_img, brightness=0, contrast=0):
        # found this function on stack overflow. All I know is that it works
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow

            buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
        else:
            buf = input_img.copy()

        if contrast != 0:
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)

            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf

    def flagNavigation(self, depth_image, color_image, showStream=False):

        cropSize = [0, 0.3]
        og_width = color_image.shape[1]
        og_height = color_image.shape[0]
        color_image = color_image[int(og_height * cropSize[1]):int(og_height * (1 - cropSize[1])),
                      int(og_width * 0):int(og_width * 1)]
        cv2.rectangle(color_image, (0, 0), (int(og_width * cropSize[0]), color_image.shape[0]), (0, 0, 0), -1)
        cv2.rectangle(color_image, (int(og_width * (1 - cropSize[0])), 0), (color_image.shape[1], color_image.shape[0]),
                      (0, 0, 0), -1)

        # cv2.imshow('1', depth_image)

        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([200, 150, 150])
        upper_blue = np.array([255, 255, 255])
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(color_image, color_image, mask=mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            self.flagCenter = -1000
            return
        biggest = [0, 0, 0, 0]
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > biggest[2] * biggest[3]:
                biggest = [x, y, w, h]

        cv2.rectangle(color_image, (biggest[0], biggest[1]), (biggest[0] + biggest[2], biggest[1] + biggest[3]),
                      (0, 255, 0), 2)

        self.flagCenter = ((biggest[0] + biggest[2] / 2) - og_width / 2) / og_width * 90

        if showStream:
            cv2.imshow('flag', color_image)

    # Create opencv window to render image in
    # cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)

    # Create colorizer object

    def personNavigate(self, img, showStream=False):
        # Convert to grayscale
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # # Detect the faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        # # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        people = self.body_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in people:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display
        if showStream:
            cv2.imshow('img', img)

    def eStop(self, depth_image, color_image, depth_color_image, showStream=False):
        self.positionEstimation(depth_image, depth_color_image, color_image, showStream)
        return

    def rowNavigation(self, depth_image, color_image, showStream=False):
        explainBadData = False  # print what is wrong when it does not know where it is going
        badDataReasons = []
        # Start streaming from file

        badData = False  # assume it is good until future tests show it is not

        cropSize = [0.2, 0.3]
        og_width = depth_image.shape[1]
        og_height = depth_image.shape[0]
        depth_image = depth_image[int(og_height * cropSize[1]):int(og_height * (1 - cropSize[1])),
                      int(og_width * 0):int(og_width * 1)]
        cv2.rectangle(depth_image, (0, 0), (int(og_width * cropSize[0]), depth_image.shape[0]), (0, 0, 0), -1)
        cv2.rectangle(depth_image, (int(og_width * (1 - cropSize[0])), 0), (depth_image.shape[1], depth_image.shape[0]),
                      (0, 0, 0), -1)

        if cv2.countNonZero(depth_image) / depth_image.size < 0.15:
            # If less than 15% of all points are invalid, just say the it is bad
            badDataReasons += ["too many invalid points"]
            badData = True

        # Convert all black pixels to white. 65535 is 16-bit
        depth_image[depth_image == 0] = 65535

        # show step

        depth_image = self.apply_brightness_contrast(depth_image, 10, 100)  # increase contrast

        # show with increased contrast
        if showStream and 2 in self.stepsShown:
            cv2.imshow('2', depth_image)

        depth_image = (depth_image / 256).astype('uint8')  # convert to 8-bit

        #  mask = np.zeros(depth_image.shape[:2], dtype="uint8")
        # cv2.rectangle(mask, (0, int(depth_image.shape[0] / 6)), (int(depth_image.shape[1]), int(depth_image.shape[0] / 1.5)), 255, -1)

        # convert all white (invalid) to  black
        depth_image[depth_image == 255] = 0
        depth_image[depth_image > 55] = 255
        depth_image[depth_image < 56] = 0

        if showStream and 3 in self.stepsShown:
            cv2.imshow('3', depth_image)

        res = depth_image.copy()

        # combine the pixel values of each column
        resized = cv2.resize(res, (res.shape[1], 1), interpolation=cv2.INTER_AREA)

        # blur it to average out outliers
        resized = cv2.blur(resized, (5, 5))

        # show the 5th step
        if showStream and 5 in self.stepsShown:
            removeThis = cv2.resize(resized, (res.shape[1], res.shape[0]), interpolation=cv2.INTER_AREA)
            cv2.imshow('5', removeThis)

        # Get the lightest colors
        x = []
        avgColor = np.mean(res)
        sd = np.std(res)

        # make the close areas black
        resized[0][:][resized[0][:] < avgColor + sd / 3] = 0

        # make the furthest points white
        b = 0
        for r in range(resized.shape[1]):
            if resized[0][r] > avgColor + sd / 3:
                b += 1
                resized[0][r] = 255
                x += [r]

        # get indices of the furthest places
        z = np.nonzero(resized[0][:])

        # the initial center is estimated as the average of all the far away points.
        frameCenter = int(depth_image.shape[1] / 2)
        centerSpot = frameCenter

        # if there actually any valid points, use the median, if not say so
        if len(z[0]) > 0:
            centerSpot = int(np.median(z))
        else:
            badDataReasons += ["no distant enough spots"]
            badData = True

        # run through the columns and find the place with the most distant points.
        # there may be distant points between stalks but they are less common and we don't want them
        # influencing the center estimation

        i = 0
        y = []
        r = np.nonzero(resized[0][:])
        numInside = 0
        numOutside = 0
        while i < len(r[0]):
            j = r[0][i]
            if centerSpot + depth_image.shape[1] / 5 > j > centerSpot - depth_image.shape[1] / 5:
                y += [j]
                resized[0][j] = 100
                numInside += 1
            else:
                numOutside += 1

            i += 1
        if numInside < numOutside * 5:
            badDataReasons += ["more lines outside than inside"]
            badData = True
        if numInside < numOutside:
            badDataReasons += ["WAY more lines outside than inside"]
            badData = True

        if len(y) > 0:
            centerSpot = int(statistics.median(y))

        if self.smoothCenter == -1000:
            self.smoothCenter = centerSpot

        if abs(centerSpot - frameCenter) > frameCenter / 3:
            badDataReasons += ["center estimated to be too far out"]
            badData = True

        # list the problems
        if explainBadData and len(badDataReasons) > 0:
            print(badDataReasons)

        # if the problem is really bad, but stick to the center of the frame
        if badData and time.time() - self.lastGood > 1 and ("WAY more lines outside than inside" in badDataReasons):
            centerSpot = frameCenter
            pass

        elif badData:  # if the estimation was invalid, use the last valid center estimation.
            centerSpot = self.smoothCenter
            badData = False

        else:  # the estimation is valid
            self.lastGood = time.time()

        # smooth the center estimation
        self.smoothCenter = (
                                        self.smoothCenter * 0 + centerSpot) / 1  # currently not smoothing center estimation, may add back later

        smoothCenterInt = int(self.smoothCenter)
        if time.time() - self.lastGood < 1:  # if there was a valid center estimation in the last second, use it.
            self.heading = (depth_image.shape[1] / 2 - self.smoothCenter) * 90 / depth_image.shape[1]
        else:
            self.heading = 1000  # set the heading to an unreasonable number so the user can know it is invalid

        if showStream:
            removeThis = cv2.resize(resized, (res.shape[1], res.shape[0]), interpolation=cv2.INTER_AREA)
            if 6 in self.stepsShown:
                cv2.imshow('6', removeThis)

            if badData:
                cv2.line(color_image, (smoothCenterInt - 2, 0),
                         (smoothCenterInt - 2, int(color_image.shape[0])), (0, 0, 255), 4)
            else:
                cv2.line(color_image, (smoothCenterInt - 2, 0),
                         (smoothCenterInt - 2, int(color_image.shape[0])), (255, 0, 0), 4)
            cv2.line(color_image, (int(color_image.shape[1] / 2), 0),
                     (int(color_image.shape[1] / 2), int(color_image.shape[0])), (0, 0, 0), 1)

        cv2.imshow('res', color_image)

        # if pressed escape exit program

    def estimate_coef(self, x, y):
        # number of observations/points
        n = np.size(x)

        # mean of x and y vector
        m_x = np.mean(x)
        m_y = np.mean(y)

        # calculating cross-deviation and deviation about x
        SS_xy = np.sum(y * x) - n * m_y * m_x
        SS_xx = np.sum(x * x) - n * m_x * m_x

        # calculating regression coefficients
        b_1 = SS_xy / SS_xx
        b_0 = m_y - b_1 * m_x

        return (b_0, b_1)

    def findString(self, color_image):

        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        lower_green = np.array([160, 100, 0])
        upper_green = np.array([170, 200, 255])
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_green, upper_green)
        # Bitwise-AND mask and original image

        res3 = cv2.bitwise_and(color_image, color_image, mask=mask)
        colorRes = res3.copy()
        res3 = cv2.cvtColor(res3, cv2.COLOR_BGR2GRAY)

        res3[res3 < 100] = 0

        # edges = cv2.bitwise_not(cv2.Canny(res3,100,200))

        h = res3.shape[1]
        w = res3.shape[0]
        res3 = res3[int(w * 0.7)::, int(h * 0.2):int(h * 0.8)]
        resized = cv2.resize(res3, (res3.shape[1], 1), interpolation=cv2.INTER_AREA)
        x = resized.copy()

        resized = cv2.resize(x, (res3.shape[1], res3.shape[0]), interpolation=cv2.INTER_AREA)

        blank = np.zeros((colorRes.shape[0], colorRes.shape[1]), np.uint8)
        j = int(w * 0.7)
        lastI = -1
        xx = np.array([])
        yy = np.array([])
        difs = np.array([])
        n = 0
        while n < len(res3):
            i = res3[n]
            if len(np.nonzero(i)[0]) > 0:
                #  print(np.median(np.nonzero(i)))
                li = int(np.median(np.nonzero(i))) + int(h * 0.2)

                if abs(li - lastI) < 10 or lastI == -1:
                    difs = np.append(difs, lastI - li)
                    lastI = li
                    #       cv2.rectangle(colorRes, (lastI, j), (lastI+1, j+1),(0, 255,0),3)
                    # print(lastI, j)
                    xx = np.append(xx, lastI)
                    yy = np.append(yy, j)
                else:
                    cv2.rectangle(blank, (li, j), (li + 1, j + 1), (0, 0, 255), 3)
                #  cv2.rectangle(colorRes, (int(xx[i]), int(yy[i])), (int(xx[i]+1), int(yy[i]+1)),(255, 0,0),3)

                # print("xx", xx)
                # print("yy", yy)
                lastI = li

            j += 3
            n += 3

        correlation_matrix = np.corrcoef(xx, yy)
        correlation_xy = correlation_matrix[0, 1]
        r_squared = correlation_xy ** 2
        # print(r_squared)
        # stdev = np.std(xx)
        # med = np.median(xx)
        # i=0
        b = self.estimate_coef(xx, yy)
        deleted = 0
        errors = []
        i = 0
        if len(xx) < 10:
            cv2.imshow('1color', blank)
            return

        i = 0
        while i < len(xx):
            cv2.rectangle(blank, (int(xx[i]), int(yy[i])), (int(xx[i] + 1), int(yy[i] + 1)), (255, 255, 255), 3)
            cv2.rectangle(color_image, (int(xx[i]), int(yy[i])), (int(xx[i] + 1), int(yy[i] + 1)), (255, 255, 255), 3)
            i += 1

            #        i+=1
        #  print(deleted)
        # print(xx, yy)
        # y = mx + b
        # x = (y-b)/m
        caan = cv2.Canny(blank, 100, 200)
        linesP = cv2.HoughLinesP(caan, 1, np.pi / 180, 50, None, 50, 10)

        cv2.imshow('2color', caan)
        bs = []
        ms = []
        maxSlope = [0, 0]
        if linesP is not None:
            print(len(linesP))
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                # cv2.line(colorRes, (l[0], l[1]), (l[2], l[3]), 255, 1, cv2.LINE_AA)
                b = self.estimate_coef(np.array([l[0], l[2]]), np.array([l[1], l[3]]))
                #    cv2.line(colorRes,(int((w-b[0])/b[1]), int(w)), (int((w/2-b[0])/b[1]), int(w/2)),(255, 0,0), 3)
                if abs(b[1]) > abs(maxSlope[1]):
                    maxSlope = b
            if abs(maxSlope[1]) > 2:
                b = maxSlope
                cv2.line(color_image, (int((w - b[0]) / b[1]), int(w)), (int((w / 2 - b[0]) / b[1]), int(w / 2)),
                         (0, 255, 0), 3)

        cv2.imshow('1color', color_image)

    #  cv2.imshow('canny', edges)

    def stalkNavigation(self, depth_image, color_image, showStream=False, colorJetMap=""):
        cv2.imshow('original', color_image)

        edges = cv2.bitwise_not(
            cv2.Canny(colorJetMap[int(colorJetMap.shape[0] / 2)::, 0:colorJetMap.shape[1]], 10, 100))
        edges2 = cv2.bitwise_not(
            cv2.Canny(color_image[int(color_image.shape[0] / 2)::, 0:color_image.shape[1]], 10, 100))

        # linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 500, None, 500, 10)
        # detector = cv2.SimpleBlobDetector()

        # keypoints = detector.detect(color_image)
        # blank = np.zeros((1,1))

        # blobs = cv2.drawKeypoints(color_image, keypoints, blank, (0,255,255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)

        # if linesP is not None:
        #     print(len(linesP))
        #     for i in linesP:
        #         i = i[0]
        #  #       print(i)

        #         cv2.line(edges,(i[0], i[1]), (i[2], i[3]), (0, 255,0), 3)
        cv2.imshow('edges', edges)
        cv2.imshow('edges2', edges2)
        # dilatation_size = cv2.getTrackbarPos(title_trackbar_kernel_size, title_dilation_window)
        # dilation_shape = morph_shape(cv2.getTrackbarPos(title_trackbar_element_shape, title_dilation_window))
        # element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
        #                                 (dilatation_size, dilatation_size))

    #    kernel = np.ones((1,1),np.uint8)
    #  dilatation_dst = cv2.dilate(edges, kernel)

    def stalkNavigationOriginal(self, depth_image, color_image, showStream=False, colorJetMap=""):

        #    self.findString(color_image)
        # cv2.imshow('depth', depth_image)
        # if not self.useCamera and self.playbackSpeed<1:
        #     time.sleep(0.1)

        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        lower_green = np.array([30, 0, 50])
        upper_green = np.array([80, 180, 180])
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_green, upper_green)
        # Bitwise-AND mask and original image

        res2 = cv2.bitwise_and(color_image, color_image, mask=mask)
        # contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # biggest = [0,0,0,0]

        edges = cv2.bitwise_not(cv2.Canny(color_image, 100, 200))

        cv2.imshow('caffnny', res2)
        res = cv2.bitwise_and(edges, edges, mask=mask)
        blur = cv2.blur(res, (5, 5))
        blur[blur < 250] = 0

        cl = color_image.copy()

        kernel = np.ones((2, 2), np.uint8)
        blur = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
        cv2.imshow('blur', blur)

        blank = np.zeros((blur.shape[0], blur.shape[1]), np.uint8)

        contours, hierarchy = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        biggest = [0, 0, 0, 0]
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > w * 2 and w < 30 and w * h > 100:
                cv2.rectangle(blank, (x, y), (x + w, y + h), 255, 3)
                cv2.rectangle(colorJetMap, (x + 40, y), (x + 40 + w, y + h), (0, 255, 0), 3)

        resized = cv2.resize(blank, (res.shape[1], 1), interpolation=cv2.INTER_AREA)
        x = resized.copy()
        x[x > 20] = 255
        x[x < 255] = 0
        resized = cv2.resize(x, (blank.shape[1], blank.shape[0]), interpolation=cv2.INTER_AREA)
        #            cv2.imshow('kd', colorJetMap)
        #     dst = cv2.Canny(blur, 50, 200, None, 3)

        #     cv2.imshow('a', blur)
        #   #  cv2.imshow('color', color_image)

        #     cl = color_image.copy()

        #      stalks = cv2.bitwise_and(colorJetMap, colorJetMap, mask= resized)
        # mask = cv2.inRange(hsv, lower_green, upper_green)
        # stalks = cv2.bitwise_and(stalks, stalks, mask= mask)

        # alternate between masked and unmasked places
        centers = []
        lastVal = 0
        n = 0
        lastOn = 0
        for i in resized[0]:
            if i == 0 and lastVal != 0 and lastOn != 0:
                centers += [[lastOn, n]]
            elif i != 0 and lastVal == 0:

                lastOn = n

            lastVal = i
            n += 1

        w = cl.shape[1]
        for c in centers:
            i = int((c[0] + c[1]) / 2)
            if abs(i - w / 2) > w / 10:
                cv2.line(cl, (i - 2, 0), (i - 2, int(cl.shape[0])), (255, 0, 0), 4)

        #  cv2.imshow('lines', cdstP)
        cv2.imshow('color', cl)

    #         cv2.imshow('stalk', resized)

    #     cv2.imshow('og_jet', depth_color_image)

    #     depth_color_image[0::, 0:depth_color_image.shape[1]-40] = depth_color_image[0::, 40::]
    #     depth_color_image[0::, depth_color_image.shape[1]-40::] = (0,0,0)
    #    # jet = cv2.bitwise_and(depth_color_image, depth_color_image, mask = resized)

    # depth = depth_image.copy()

    # depth = self.apply_brightness_contrast(depth, 10,100)  # remove this line

    # depth[0::, 0:depth.shape[1]-40] = depth[0::, 40::]
    # depth[0::, depth.shape[1]-40::] = 0
    # jet = cv2.bitwise_and(depth, depth, mask = resized)

    # depth_regress = np.zeros((depth.shape[0],depth.shape[1]),np.uint8)

    # for i in centers:
    #     i[0]-=40
    #     i[1]-=40
    #     a = np.nonzero(depth[0::, i[0]:i[1]])
    #     if len(a)>0:
    #         x = np.median(a)

    #         depth_regress[0::, i[0]:i[1]][True] = x

    #    cv2.imshow('jet', depth_regress)

    # w = blur.shape[1]
    # h = blur.shape[0]
    # rampl = np.linspace(1, 0, int(w/2))
    # rampl = np.tile(np.transpose(rampl), (h,1))
    # rampl = cv2.merge([rampl,rampl,rampl])

    # vis = np.concatenate((cv2.flip(rampl,1), rampl), axis=1)
    # cv2.imshow('edgddes',vis)
    # blur[depth_image < vis] = 100

    #   edges = cv2.blur(edges, (20, 20))

    # for cnt in contours:
    #     x,y,w,h = cv2.boundingRect(cnt)
    #     if w*h>biggest[2]*biggest[3]:
    #         biggest = [x,y,w,h]
    #         cv2.rectangle(color_image,(biggest[0],biggest[1]),(biggest[0]+biggest[2], biggest[1]+biggest[3]),(0,255,0),2)

    # cv2.rectangle(color_image,(biggest[0],biggest[1]),(biggest[0]+biggest[2], biggest[1]+biggest[3]),(0,255,0),2)

    # cv2.imshow('edges',edges)

    #    cv2.imshow('res',res)

    # cv2.imshow('og', color_image)

    def aboveNavigate(self, img, showStream=False):
        #    return
        if showStream:
            cv2.imshow('Frame', img)

    def videoNavigate(self, navTypes, showStream=False):
        self.cap = False
        self.savedVideo = False
        if True:
            profile = self.pipeline.start(self.config)

            if not self.useCamera and (not self.realtime or self.startFrame > 0):
                playback = profile.get_device().as_playback()
                playback.set_real_time(False)

            colorizer = rs.colorizer()

            # start time
            self.lastGood = time.time()
            waitKeyType = 0
            frameCount = 0

        print("started video pipeline")

        if "above" in navTypes or "person" in navTypes:
            if self.rgbFilename != "" and not self.useCamera:
                self.cap = cv2.VideoCapture(self.rgbFilename)
            else:
                self.cap = cv2.VideoCapture(0)
            self.face_cascade = cv2.CascadeClassifier(
                '/home/john/Desktop/ROS_field_robot/haarcascade_frontalface_default.xml')
            self.body_cascade = cv2.CascadeClassifier('/home/john/Desktop/ROS_field_robot/haarcascade_fullbody.xml')

            if (self.cap.isOpened() == False):
                print("Error reading video file")
                return
            if self.saveVideo and self.useCamera:
                frame_width = int(self.cap.get(3))
                frame_height = int(self.cap.get(4))

                size = (frame_width, frame_height)

                self.savedVideo = cv2.VideoWriter(self.rgbFilename, cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

        # Streaming loop
        while not self.stop:

            #      print("getting frame")
            # Get frameset of depth
            while frameCount < self.startFrame:
                frames = self.pipeline.wait_for_frames()
                frameCount += 1

            if True:
                frames = self.pipeline.wait_for_frames()
                frameCount += 1
                # Get depth frame
                depth_frame = frames.get_depth_frame()

                # if not self.useCamera and self.playbackSpeed<1:
                #     time.sleep(0.1)

                #
            #    if showStream:
            # the jetmap color image is never used so don't bother creating it unless you need to show it

            # Colorize depth frame to jet colormap

            #

            depth_color_frame = colorizer.colorize(depth_frame)

            depth_color_image = np.asanyarray(depth_color_frame.get_data())
            if showStream and 1 in self.stepsShown:
                cv2.imshow('1', depth_color_image)
            #       cv2.imshow('133', depth_color_image)

            rgb_frame = frames.get_color_frame()
            color_image = np.asanyarray(rgb_frame.get_data())
            #    cv2.imshow('2', color_image)

            #  Convert depth_frame to numpy array to render image in opencv
            depth_image = np.asanyarray(depth_frame.get_data())

            if "row" in navTypes:
                self.rowNavigation(depth_image.copy(), color_image.copy(), showStream)
            if "flag" in navTypes:
                self.flagNavigation(depth_image.copy(), color_image.copy(), showStream)
            if "stalk" in navTypes:
                self.stalkNavigation(depth_image.copy(), color_image.copy(), showStream, depth_color_image)
            if "eStop" in navTypes:
                self.eStop(depth_image.copy(), color_image.copy(), depth_color_image.copy(), showStream)
            if ("above" in navTypes or "person" in navTypes) and self.cap.isOpened():
                ret, frame = self.cap.read()

                if ret == True:
                    #  print("good")
                    if self.saveVideo and self.useCamera:
                        self.savedVideo.write(frame)
                    if "above" in navTypes:
                        self.aboveNavigate(frame, showStream=showStream)
                    if "person" in navTypes:
                        self.personNavigate(frame, showStream=showStream)
                else:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            if waitKeyType == 32:
                key = cv2.waitKey(0)
            else:
                key = cv2.waitKey(1)
            waitKeyType = key
            if key == 27:
                cv2.destroyAllWindows()
                self.stopStream()
                break

    def stopStream(self):

        self.heading = -1000
        self.flagCenter = -1000
        if self.cap != False:
            self.cap.release()
        if self.savedVideo != False:
            self.savedVideo.release()

        if not self.stop:
            self.stop = True
            try:
                self.pipeline.stop()
                print("pipeline stopped")
            except:
                print("pipeline already stopped")


if __name__ == "__main__":
    cam = RSCamera(_useCamera, _saveVideo, filename=_filename, rgbFilename=_rgbFilename, realtime=_realtime,
                   startFrame=_startFrame, stepsShown=_stepsShown)
    cam.videoNavigate(_navTypes, _showStream)
