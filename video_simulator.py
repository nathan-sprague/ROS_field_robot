import time

import cv2

import numpy as np

import math

import random



class ObjShape:
    def __init__(self, pts, triangles, color = (0,0,255)):

        self.pts = np.array(pts).transpose() # numpy array format ((x1, x2, x3...) (y1,y2,y3...) (z1,z2,z3...))
        self.triangles = np.array(triangles) # numpy array format ((pt1, pt2, pt3), (pt1,pt2,pt3)...)

        self.color = color

        # upper and lower limits to avoid calculating every point to see the layer number of the leaf
        self.limits = ((np.min(self.pts[0]), np.max(self.pts[0])), (np.min(self.pts[1]), np.max(self.pts[1])), (np.min(self.pts[2]), np.max(self.pts[2])))
        self.lowResTriangles = np.array([[0,1,2]])
        self.lowResPts = np.array([np.random.choice(self.pts[0], 3), np.random.choice(self.pts[1], 3), np.random.choice(self.pts[2], 3)])
            # [np.min(self.pts[0]), np.min(self.pts[1]), np.min(self.pts[2])], 
            # [np.max(self.pts[0]), np.min(self.pts[1]), np.min(self.pts[2])],
            # [np.min(self.pts[0]), np.max(self.pts[1]), np.max(self.pts[2])]  ] ).transpose()

        # print(self.lowResPts)
        # print("pts real", self.pts)
        # print()

        # print(self.lowResTriangles)
        # print("triangle real", self.triangles)
        # exit()

class VideoSim:
    def __init__(self, freeMove = False, rows = []):
        """
        generates a simulated video of corn
        """

        print("setting up video simulation")

        self.freeMove = freeMove
        self.rows = rows

        self.objList = []
        self.drawPts = False
        self.drawLines = False
        self.ordered = True
        self.ht = 480
        self.wt = 640
     
        self.projectionFactor = 12 # ft to inches

        self.floor = self.makeEmptyFloor() # make an empty floor as the base for depth

        self.begin()



    def begin(self):
        """
        build map
        if in free movement mode, go into main loop

        """
        

        # make rows of corn
        totalPts = 0
        numPlants = 0
        i=-3
        while i<3:
            j=-200+(abs(i)*10)
            while j < 0:

                # make the projection thing manually. I tried pyproj but it didn't work quite right
                coords = [40.471797, -86.995246] # starting point from acre bay far north destination. Find a way to make it work for any destination later.
                x = coords[1] * 364000
                longCorrection = math.cos(coords[0] * math.pi / 180)
                y = coords[0] * longCorrection * 364000
                x*=self.projectionFactor
                y*=self.projectionFactor
                totalPts += self.makeCorn(x+j*4, y+i*30+0, 1.3, 60, 3)# 30-inch rows, 4 inches apart
                numPlants += 1
                 
                j+=1 

        

            i+=1
        print("generated", numPlants, "plants totaling", int(totalPts/3), "triangles")


        if self.freeMove: # allow the user to move the camera manually
            # x,y =coords
            cameraView = [1.6, 0]

            cameraLocation = [x+1, y-30, 12]
            key = 0

            while key != 27:

                depth_image, color_image = self.draw3D(cameraView, cameraLocation)
                cv2.imshow("free depth", depth_image)
                cv2.imshow("free color", color_image)
                
                key = cv2.waitKey(1)
                speed = 1
                if key == 82: # up key
                    cameraLocation[0] += math.sin(cameraView[0]) * -speed
                    cameraLocation[1] += math.cos(cameraView[0]) * -speed 
                elif key == 84: # back key
                    cameraLocation[0] += math.sin(cameraView[0]) * speed
                    cameraLocation[1] += math.cos(cameraView[0]) * speed
                elif key == 83: # right key
                    cameraLocation[0] += math.cos(cameraView[0]) * -speed
                    cameraLocation[1] += math.sin(cameraView[0]) * speed
                elif key == 81: # left key
                    cameraLocation[0] += math.cos(cameraView[0]) * speed
                    cameraLocation[1] += math.sin(cameraView[0]) * -speed
                elif key == 32: # space
                    cameraLocation[2] += 1
                elif key == 225: # space
                    cameraLocation[2] -= 1

                elif key == 97: # a
                    cameraView[0] -= 0.1
                elif key == 100: # d
                    cameraView[0] += 0.1
                elif key == 119: # w
                    cameraView[1] += 0.1
                elif key == 115: # s
                    if abs(cameraView[1])<math.pi/3:
                        cameraView[1] -= 0.1

                elif key != -1 and key != 27: # unknown character
                    print("unknown key:", key)
                # print(key)

                time.sleep(0.1)
                i+=1

    def makeEmptyFloor(self):
        """
        make simulated empty depth floor.
        Returns a 16 bit image, like the realsense image type.
        """

        floorHorizon = 240
        horizonMax = 7000
        rampr16 = np.zeros((480, 1)).astype('uint16')

        # print(floorHorizon, horizonMax)

        i=0
        val = 0
        if horizonMax < 1:
            horizonMax = 1
        factor = 0.5

        while i<480:
            if i < 100:
                val=250*256
            elif i > floorHorizon:
                val=(2**(-(i-floorHorizon)/30)*12+2) * 256
            
            rampr16[i][0] = int(val / factor)
            i+=1
        rampr16 = cv2.resize(rampr16, (640, 480), interpolation=cv2.INTER_AREA)
        rampr16 = rampr16
        return rampr16



    def makeCorn(self, x, y, width, height, resolution):
        color =(50*random.random(), (50*random.random()+ 130), 20*random.random())
        numLeaves = 12

        # shift the stalk location randomly
        x+= width*random.random()*2
        y+= width*random.random()*2

        # make the stalks
        i=0
        step = 2*math.pi/resolution;
        angle = 0;
        stalk = [];
        stalkTriangles = [];

        if resolution == 2: # resolution of 2 is the same as 1, but with more points
            i=1

        while (i<resolution):
            # make a bunch of triangles for the stalk
            stalk+=[[x+math.sin(angle)*width/2, y+math.cos(angle)*width/2, 0]];
            stalk+=[[x+math.sin(angle+step)*width/2, y+math.cos(angle+step)*width/2, 0]];
            stalk+=[[x+math.sin(angle)*width/3, y+math.cos(angle)*width/3, height]];
            stalk+=[[x+math.sin(angle+step)*width/3, y+math.cos(angle+step)*width/3, height]];
            cl = len(stalk)
            stalkTriangles+=[[cl-4, cl-3, cl-2], [cl-3, cl-2, cl-1]];
            angle += step;
            i+=1;
        self.objList+=[ObjShape(stalk, stalkTriangles, color)]


        # make the leaves
        leafStartHeight = (0.05 + random.random()*0.1) # as a fraction of full height. This is random, change as needed
        leafStep = (1-leafStartHeight)/numLeaves # even spacing to the top
        leafWidth = width
        leafAngle = random.random()*math.pi
        i=0
        leafPts = []
        leafTriangles = []

        while i < numLeaves-2: # top 2 leaves are useless
            l = 0

            leafAngle += math.pi + random.random()*0.1 # alternate sides

            leafHeight = leafStartHeight + leafStep*i

            leafLength = (1.3-(2*leafHeight-1)**2)/0.5 # some random equation to get a upside down parabola
            leafLength *= height/5

            j=0
            while j<resolution:
                h1 = leafLength*(-(j/resolution-0.8)**2+1) + leafHeight*height  # another random upside down parabola
                x1 = x+math.sin(leafAngle)*j*leafLength/resolution
                y2 = y+math.cos(leafAngle)*j*leafLength/resolution
                leafPts+=[[x1-leafWidth, y2-leafWidth, h1]]
                leafPts+=[[x1+leafWidth, y2+leafWidth, h1]]

                j+=1
                h2 = leafLength*(-(j/resolution-0.8)**2+1)+ leafHeight*height
                x2 = x+math.sin(leafAngle)*j*leafLength/resolution
                y2 = y+math.cos(leafAngle)*j*leafLength/resolution
                leafPts+=[[x2-leafWidth, y2-leafWidth, h2]]
                leafPts+=[[x2+leafWidth, y2+leafWidth, h2]]

                cl = len(leafPts)
                leafTriangles+=[[cl-4, cl-3, cl-2], [cl-3, cl-2, cl-1]]
                # j+=1
                



            i += 1
        self.objList+=[ObjShape(leafPts, leafTriangles, color)]

        return len(stalk) + len(leafPts)



    def draw3D(self, cameraView, cameraLocation):


        # set up blank images
        color_image = np.zeros((self.ht, self.wt, 3)).astype('uint8')
        color_image[:] = (255, 180, 180)

        # give the color image a ground
        groundColor = (15, 60, 120)
        horiz = int(math.sin(cameraView[1])/math.cos(cameraView[1])*self.wt/2+self.ht/2);
        if (horiz>0):
            cv2.rectangle(color_image, (0, horiz), (self.wt, self.ht), groundColor, -1);
        else:
            cv2.rectangle(color_image, (0, 0), (self.wt, self.ht), groundColor, -1);
        # print(horiz)

        depth_image = self.floor.copy() # np.zeros((self.ht, self.wt)).astype('uint16')

        return depth_image, color_image # DELETE THIS IF YOU ACTUALLY WANT VIDEO!!!!!!!


        objsToUse = [];
        trianglesMade = 0


        # constants to rotate the points
        s1 = math.sin(cameraView[0])
        c1 = math.cos(cameraView[0])
        s2 = math.sin(cameraView[1])
        c2 = math.cos(cameraView[1])

        for i in self.objList: # rotate the object's limits and check if they are in sight of the camera
            minPt = self.rotatePt3D(i.limits[0][0], i.limits[1][0], i.limits[2][0], s1, c1, s2, c2, cameraLocation)
            maxPt = self.rotatePt3D(i.limits[0][1], i.limits[1][1], i.limits[2][1], s1, c1, s2, c2, cameraLocation)

            
            if minPt[1] < 0 or maxPt[1] < 0: # in sight of the camera, add to the list of objects to draw

                d = (cameraLocation[0]-i.limits[0][0])**2 + (cameraLocation[1]-i.limits[1][0])**2 + (cameraLocation[2]-i.limits[2][0])**2 # squared distance from camera
                if d < 90000:
                    objsToUse += [[(maxPt[1]+minPt[1])/2, i, d]]
                

        

        objsToUse.sort(key = lambda x: x[0])
        # print("checking", len(self.objList), "objs, made", len(objsToUse))


        for i in objsToUse:

            if False:# i[2] > 20000:
                triangles = i[1].lowResTriangles
                pts = i[1].lowResPts
            else:
                triangles = i[1].triangles
                pts = i[1].pts

            # get the relative coordinates
            x = pts[0]-cameraLocation[0]
            y = pts[1]-cameraLocation[1]
            z = pts[2]-cameraLocation[2]


            # left/right rotation
            xRot = x * c1 - y * s1
            yRot = x * s1 + y * c1

            # up/down rotation
            zRot = yRot * s2 + z * c2
            yRot = yRot * c2 - z * s2

    
            x = xRot / yRot * self.wt/2 + self.wt/2;
            y = zRot / yRot * self.wt/2 + self.ht/2;
            z = yRot

            # triangle coords
            triX = x[triangles]
            triY = y[triangles]
            triZ = z[triangles]

            j=0
            while j<len(triZ):
 
                if max(triZ[j]) > 0:
                    j+=1
                else:
                    poly = np.array((triX[j], triY[j]), np.int32).transpose()
                    poly = poly.reshape((-1, 1, 2))

                    dist = -np.mean(triZ[j])
                    
                    cv2.fillPoly(depth_image, [poly], int(dist)*40)
                    cv2.fillPoly(color_image, [poly], i[1].color)
                    trianglesMade += 1
                    j+=1

        depth_image = cv2.blur(depth_image, (5, 5)) # blur it a bit becasue the IRL depth isn't great
        # print("made", trianglesMade, "triangles")

        return depth_image, color_image

     



    def rotatePt3D(self, x, y, z, s1, c1, s2, c2, cameraLocation):
        # rotates a point in 3 dimensions. s1,c1 etc are sin/cos of the camera angle/rotation amount

        x = x-cameraLocation[0]
        y = y-cameraLocation[1]
        z = z-cameraLocation[2]

        # left/right rotation
        xRot = x * c1 - y * s1
        yRot = x * s1 + y * c1

        # up/down rotation
        zRot = yRot * s2 + z * c2
        yRot = yRot * c2 - z * s2

        return [xRot,yRot,zRot]





if __name__ == "__main__":

    startRow = (40.4718345, -86.9952429)
    rows = []
    i = 0
    while i < 10:
        rows += [[[startRow[0]-i*0.00001, startRow[1]], [startRow[0]-i*0.00001, startRow[1]-0.0007]]]
        i+=1


    cam = VideoSim(freeMove = True, rows = rows)

