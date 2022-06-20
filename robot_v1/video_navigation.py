from numpy.core.defchararray import center
import pyrealsense2 as rs
import statistics


import numpy as np
import cv2
import argparse  # Import argparse for command-line options
import os.path
import math
import time

_stepsShown =  [1, 2, 3, 6]

# _filename = "Desktop/real_corn_july_15/human.bag"
_filename = "deleteme.bag"#"object_detections/real_corn_july_28/outside_row_long.bag"
# _filename = "object_detections/real_corn_july_28/exit_row.bag"
# _filename = "Desktop/fake_track/object_detection6.bag"
# _filename=""
_useCamera = True
_showStream = True
_saveVideo = False
_realtime = True
_startFrame = 0
_rgbFilename = "blah.mp4"
_navTypes = ["above", "row"]

class RSCamera:
    def __init__(self, useCamera, saveVideo, filename="", rgbFilename="", realtime=True, startFrame=0, stepsShown = []):
        self.useCamera = useCamera
        self.saveVideo = saveVideo
        self.realtime = realtime
        self.startFrame = startFrame
        self.stop = False
        self.heading = 0
        self.smoothCenter = -1000
        self.flagCenter = -1000
        self.stepsShown = stepsShown

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

        else:
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
            color_image = color_image[int(og_height*cropSize[1]):int(og_height*(1-cropSize[1])), int(og_width*0):int(og_width*1)]
            cv2.rectangle(color_image, (0,0), (int(og_width*cropSize[0]), color_image.shape[0]), (0,0,0), -1)
            cv2.rectangle(color_image, (int(og_width*(1-cropSize[0])),0), (color_image.shape[1], color_image.shape[0]), (0,0,0), -1)

           # cv2.imshow('1', depth_image)
     

            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

            lower_blue = np.array([100,150,150])
            upper_blue = np.array([255,255,255])
            # Threshold the HSV image to get only blue colors
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            # Bitwise-AND mask and original image
            res = cv2.bitwise_and(color_image, color_image, mask= mask)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                self.flagCenter = -1000
                return
            biggest = [0,0,0,0]
            for cnt in contours:
                x,y,w,h = cv2.boundingRect(cnt)
                if w*h>biggest[2]*biggest[3]:
                    biggest = [x,y,w,h]
            
            cv2.rectangle(color_image,(biggest[0],biggest[1]),(biggest[0]+biggest[2], biggest[1]+biggest[3]),(0,255,0),2)

            self.flagCenter = ((biggest[0]+biggest[2]/2) - og_width/2) / og_width * 90

            if showStream:
                cv2.imshow('flag', color_image)

                    


    # Create opencv window to render image in
    # cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)

    # Create colorizer object
    
    def rowNavigation(self, depth_image, color_image, showStream=False):
        explainBadData = False  # print what is wrong when it does not know where it is going
        badDataReasons = []
        # Start streaming from file
        

        badData = False  # assume it is good until future tests show it is not

        
        cropSize = [0.2, 0.3]
        og_width = depth_image.shape[1]
        og_height = depth_image.shape[0]
        depth_image = depth_image[int(og_height*cropSize[1]):int(og_height*(1-cropSize[1])), int(og_width*0):int(og_width*1)]
        cv2.rectangle(depth_image, (0,0), (int(og_width*cropSize[0]), depth_image.shape[0]), (0,0,0), -1)
        cv2.rectangle(depth_image, (int(og_width*(1-cropSize[0])),0), (depth_image.shape[1], depth_image.shape[0]), (0,0,0), -1)


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

        elif badData: # if the estimation was invalid, use the last valid center estimation.
            centerSpot = self.smoothCenter
            badData = False

        else: # the estimation is valid
            self.lastGood = time.time()

        # smooth the center estimation
        self.smoothCenter = (self.smoothCenter * 0 + centerSpot) / 1  # currently not smoothing center estimation, may add back later
        
        

        smoothCenterInt = int(self.smoothCenter)
        if time.time() - self.lastGood < 1: # if there was a valid center estimation in the last second, use it.
            self.heading = (depth_image.shape[1] / 2 - self.smoothCenter) * 90 / depth_image.shape[1]
        else:
            self.heading = 1000 # set the heading to an unreasonable number so the user can know it is invalid

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
            

    def stalkNavigation(self, depth_image, color_image, showStream = False, colorJetMap=""):


           # cv2.imshow('depth', depth_image)
            # if not self.useCamera and self.playbackSpeed<1:
            #     time.sleep(0.1)
            

        
            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

            lower_green = np.array([30,0,50])
            upper_green = np.array([80,180,180])
            # Threshold the HSV image to get only blue colors
            mask = cv2.inRange(hsv, lower_green, upper_green)
            # Bitwise-AND mask and original image
            
           # res2 = cv2.bitwise_and(depth_color_image, depth_color_image, mask= mask)
            # contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # biggest = [0,0,0,0]
         
            

            edges = cv2.bitwise_not(cv2.Canny(color_image,100,200))
  
            cv2.imshow('canny', edges)
            res = cv2.bitwise_and(edges, edges, mask= mask)
            blur = cv2.blur(res, (5, 5))
            blur[blur < 250] = 0

       
        # #    depth_image = cv2.flip(depth_image, 1)
        #     d_copy = cv2.transpose(depth_image)
        #     w = d_copy.shape[0]
        #     maximum = 1
        #     for i in range(0, w):
        #         a = d_copy[i]
        #      #   print(a)
             
        #         amount = maximum-abs(maximum-i/w*maximum*2)
        #         a[a*255/65535 > amount] = 65535
        #         a[a == 0] = 65535
        #         # if amount>maximum/2:
        #         #     a[a*255/65535 > -1] = 65535
        #         # if amount>maximum/1.5:
        #         #     a[a*255/65535 > -1] = int(65535/2)
        #      #   

        #         #print(a)
        #         d_copy[i] = a
        #      #   print(i)
        #     d_copy = cv2.transpose(d_copy)
        #     d_copy = cv2.flip(d_copy,1)

        #     d_copy = (depth_image / 256).astype('uint8')

        #     d_copy = cv2.bitwise_and(blur, blur, mask= d_copy)

            cl = color_image.copy()
            print(depth_image[int(depth_image.shape[0]/2)])
            cv2.imshow('assdfd2', depth_image)
         #   cv2.imshow('depth', depth_image)
            depth_image2 = self.apply_brightness_contrast(depth_image, 10, 100)  # increase contrast

            # show with increased contrast
            depth_image2[depth_image2==0] = 65500
            cv2.imshow('asfd2', depth_image2)

            kernel = np.ones((2,2),np.uint8)
            blur = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
            cv2.imshow('blur', blur)


            blank = np.zeros((blur.shape[0],blur.shape[1]),np.uint8)

            contours, hierarchy = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            biggest = [0,0,0,0]
            for cnt in contours:
                x,y,w,h = cv2.boundingRect(cnt)
                if h>w*2 and w<30 and w*h>100:
                    cv2.rectangle(blank,(x,y),(x+w, y+h),255,3)
                    cv2.rectangle(colorJetMap,(x+40,y),(x+40+w, y+h),(0, 255,0),3)

        
            
            resized = cv2.resize(blank, (res.shape[1], 1), interpolation=cv2.INTER_AREA)
            x = resized.copy()
            x[x>20] = 255
            x[x<255] = 0
            resized = cv2.resize(x, (blank.shape[1], blank.shape[0]), interpolation=cv2.INTER_AREA)
            cv2.imshow('kd', colorJetMap)
        #     dst = cv2.Canny(blur, 50, 200, None, 3)

        #     cv2.imshow('a', blur)
        #   #  cv2.imshow('color', color_image)

        #     cl = color_image.copy()

            stalks = cv2.bitwise_and(colorJetMap, colorJetMap, mask= resized)
            #mask = cv2.inRange(hsv, lower_green, upper_green)
            #stalks = cv2.bitwise_and(stalks, stalks, mask= mask)

   
            centers = []
            
            lastVal = 0
            n = 0
            lastOn = 0
            
            for i in resized[0]:
                if i == 0 and lastVal!= 0 and lastOn!=0:
                    centers += [[lastOn, n]]
                elif i!=0 and lastVal == 0:
                    
                    lastOn = n

                lastVal = i
                n+=1
            

            
            w=cl.shape[1]
            for c in centers:
                i = int((c[0]+c[1])/2)
                if abs(i-w/2) > w/10:
                    cv2.line(cl, (i - 2, 0), (i - 2, int(cl.shape[0])), (255, 0, 0), 4)

            

            # linesP = cv2.HoughLinesP(resized, 5, np.pi / 180, 50, None, 50, 10)
            
            # if linesP is not None:
            # #    print(len(linesP))
            #     for i in range(0, len(linesP)):
            #         l = linesP[i][0]
            #         deg = math.degrees(math.atan2(l[0]-l[2],l[1]-l[3]))
                
            #         if abs(deg) > 170 or abs(deg) < 10:
                        
            #             cv2.line(cl, (l[0], l[1]), (l[2], l[3]), (0,0,255), 5, cv2.LINE_AA)


          #  cv2.imshow('lines', cdstP)
            cv2.imshow('color', cl)
            cv2.imshow('stalk', stalks)

            



            
        #     cv2.imshow('og_jet', depth_color_image)
            
        
        #     depth_color_image[0::, 0:depth_color_image.shape[1]-40] = depth_color_image[0::, 40::]
        #     depth_color_image[0::, depth_color_image.shape[1]-40::] = (0,0,0)
        #    # jet = cv2.bitwise_and(depth_color_image, depth_color_image, mask = resized)


            depth = depth_image.copy()
            
            depth = self.apply_brightness_contrast(depth, 10,100)  # remove this line

            depth[0::, 0:depth.shape[1]-40] = depth[0::, 40::]
            depth[0::, depth.shape[1]-40::] = 0
            jet = cv2.bitwise_and(depth, depth, mask = resized)

            depth_regress = np.zeros((depth.shape[0],depth.shape[1]),np.uint8)

            for i in centers:
                i[0]-=40
                i[1]-=40
                a = np.nonzero(depth[0::, i[0]:i[1]])
                if len(a)>0:
                    x = np.median(a)

                    depth_regress[0::, i[0]:i[1]][True] = x




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
            

         #   cv2.imshow('og', color_image)

    def aboveNavigate(self, img, showStream = False):
        if showStream:
            cv2.imshow('Frame', img)
    


    def videoNavigate(self, navTypes, showStream = False):
        self.cap = False
        self.savedVideo = False
        profile = self.pipeline.start(self.config)

        if not self.useCamera and (not self.realtime or self.startFrame>0):
            playback=profile.get_device().as_playback()
            playback.set_real_time(False)
            
        colorizer = rs.colorizer()

        # start time
        self.lastGood = time.time()
        waitKeyType = 0
        frameCount = 0

        print("started video pipeline")

        if "above" in navTypes:
            if self.rgbFilename!="" and not self.useCamera:
                self.cap = cv2.VideoCapture(self.rgbFilename)
            else:
                self.cap = cv2.VideoCapture(0)

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
            while frameCount<self.startFrame:
                frames = self.pipeline.wait_for_frames()
                frameCount += 1
                
      

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
         
             #   if showStream and 1 in self.stepsShown:
                #    cv2.imshow('1', depth_color_image)
            depth_color_frame = colorizer.colorize(depth_frame)

            depth_color_image = np.asanyarray(depth_color_frame.get_data())
      #      cv2.imshow('133', depth_color_image)

            rgb_frame = frames.get_color_frame()
            color_image = np.asanyarray(rgb_frame.get_data())

            #  Convert depth_frame to numpy array to render image in opencv
            depth_image = np.asanyarray(depth_frame.get_data())


            if "row" in navTypes:
                self.rowNavigation(depth_image.copy(), color_image.copy(), showStream)
            if "flag" in navTypes:
                self.flagNavigation(depth_image.copy(), color_image.copy(), showStream)
            if "stalk" in navTypes:
                self.stalkNavigation(depth_image.copy(), color_image.copy(), showStream, depth_color_image)
            if "above" in navTypes and self.cap.isOpened():
             #   print("yo")
                ret, frame = self.cap.read()

                if ret == True:
                  #  print("good")
                    if self.saveVideo and self.useCamera:
                        self.savedVideo.write(frame)
                    self.aboveNavigate(frame, showStream=True)
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
    cam = RSCamera(_useCamera, _saveVideo, filename=_filename, rgbFilename=_rgbFilename, realtime=_realtime, startFrame=_startFrame, stepsShown=_stepsShown)
    cam.videoNavigate(_navTypes, _showStream)
