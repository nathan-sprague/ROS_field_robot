import time
import cv2
import numpy as np
import math



def makeGuardrail(camera, startX, startY, endX, endY, triangleCount =0, sideFacing=True):
    allRed = True

    startX, endX = min(startX, endX), max(startX, endX)
    startY, endY = min(startY, endY), max(startY, endY)
    if startX==endX:
        angle = math.pi/2
    else:
        angle = math.atan((endY-startY)/(endX-startX))


    height = 0.8
    # make the stalks
    i=0
    post = [];

    l = 0.3
    w = 0.3

    color = (30,100,130)

    if allRed:
        color = (0,0,255)


    subrail_separation = 0.1
    x = startX + math.cos(angle)*subrail_separation/2
    y = startY + math.sin(angle)*subrail_separation/2
    x2=x+math.cos(angle)*subrail_separation
    y2=y+math.sin(angle)*subrail_separation

    print(startX, endX, startY, endY)
    postCounter = 0

    if sideFacing:
        cX = -math.sin(angle)*w/2
        cY = -math.cos(angle)*w/2
    else:
        cX = math.sin(angle)*w/2
        cY = math.cos(angle)*w/2

    lastSep = -1
    while x2 <= endX and y2 <= endY:
        if postCounter - lastSep > 1.5:
            print("making post", x,y)
            post, postTriangles = camera.getCube(x, y, l, w, angle, 0, height)

            camera.entities.append(Entity(post, postTriangles, color, idStart=triangleCount+1))
            triangleCount += len(postTriangles)
            lastSep = postCounter
            # return

        rail, railTriangles = camera.getCube(x+cX, y+cY, subrail_separation, 0.1, angle, 0.4, height)

        camera.entities.append(Entity(rail, railTriangles, (100,100,100), idStart=triangleCount+1))

        postCounter+=subrail_separation
        x+=math.cos(angle)*subrail_separation
        y+=math.sin(angle)*subrail_separation

        x2=x+math.cos(angle)*subrail_separation/2
        y2=y+math.sin(angle)*subrail_separation/2



class Entity():
     def __init__(self, pts, triangles, color = (0,0,255), idStart=0):
        scale = 30 # convert to meters
        self.pts = np.array(pts).transpose() # numpy array format ((x1, x2, x3...) (y1,y2,y3...) (z1,z2,z3...))
        self.triangles = np.array(triangles) # numpy array format ((pt1, pt2, pt3), (pt1,pt2,pt3)...)
        self.triangle_colors = [color for i in range(len(triangles))]
        self.occluded = np.zeros(len(triangles))+1
        self.color = color
        self.limits = np.array([(np.min(self.pts[0]), np.max(self.pts[0])), (np.min(self.pts[1]), np.max(self.pts[1])), (np.min(self.pts[2]), np.max(self.pts[2]))])

class Camera():

    def __init__(self, free_move = False, projection_scale=30):

        print("setting up video simulation")

        self.free_move = free_move

        self.entities = []

        self.drawPts = False
        self.drawLines = False
        self.ordered = True
        self.ht = 480
        self.wt = 640


        self.lastCameraView = [-1,-1] 
        self.lastCameraLocation = [-1, -1]
     
        self.projection_scale = projection_scale#12 # ft to inches


    def makeCorn(self, x, y, width, height, resolution, triangleCount =0):
        color =(50*np.random.random(), (50*np.random.random()+ 130), 20*np.random.random())
        numLeaves = 12

        # shift the stalk location randomly
        # x+= width*random.random()*2
        # y+= width*random.random()*2

        

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
        self.entities.append(Entity(stalk, stalkTriangles, color, idStart=triangleCount+1))
        triangleCount += len(stalkTriangles)


        # make the leaves
        leafStartHeight = (0.05 + np.random.random()*0.1) # as a fraction of full height. This is random, change as needed
        leafStep = (1-leafStartHeight)/numLeaves # even spacing to the top
        leafWidth = width
        leafAngle = np.random.random()*math.pi
        i=0
        leafPts = []
        leafTriangles = []

        while i < numLeaves-2: # top 2 leaves are useless
            l = 0

            leafAngle += math.pi + np.random.random()*0.1 # alternate sides

            leafHeight = leafStartHeight + leafStep*i

            leafLength = (1.3-(2*leafHeight-1)**2)/0.5 # some random equation to get a upside down parabola
            leafLength *= height/5

            j=0
            while j<resolution:
                h1 = leafLength*(-(j/resolution-0.8)**2+1) + leafHeight*height  # another random upside down parabola
                x1 = x+math.sin(leafAngle)*j*leafLength/resolution
                y2 = y+math.cos(leafAngle)*j*leafLength/resolution
                leafPts += [[x1-leafWidth, y2-leafWidth, h1]]
                leafPts += [[x1+leafWidth, y2+leafWidth, h1]]

                j+=1
                h2 = leafLength*(-(j/resolution-0.8)**2+1)+ leafHeight*height
                x2 = x + math.sin(leafAngle)*j*leafLength/resolution
                y2 = y + math.cos(leafAngle)*j*leafLength/resolution
                leafPts += [[x2-leafWidth, y2-leafWidth, h2]]
                leafPts += [[x2+leafWidth, y2+leafWidth, h2]]

                cl = len(leafPts)
                leafTriangles+=[[cl-4, cl-3, cl-2], [cl-3, cl-2, cl-1]]
                # j+=1
                



            i += 1
        self.entities+=[Entity(leafPts, leafTriangles, color, triangleCount+1)]
        triangleCount += len(leafTriangles)

        return len(stalk) + len(leafPts)



    def getCube(self, x, y, l, w, angle, startH, endH):

        points = np.array([
            [-l / 2, w / 2],
            [l / 2, w / 2],
            [l / 2, -w / 2],
            [-l / 2, -w / 2]
        ])

        # Rotation matrix
        R = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])

        # Rotate and translate the points
        rotated_points = [np.dot(R, p) for p in points]
        # post = [[int(x + p[0]), int(y + p[1]), startH] for p in rotated_points]
        post = [[(x + p[0]), (y + p[1]), startH] for p in rotated_points]

        # post += [[int(x + p[0]), int(y + p[1]), endH] for p in rotated_points]
        post += [[(x + p[0]), (y + p[1]), endH] for p in rotated_points]

        postTriangles = [[3,0,7], [0,7,4], [2,3,6], [3,6,7], [1, 2, 5], [2,5,6], [0, 1, 4], [1,5,4], [0, 1, 2], [0, 3, 2]]

        return post, postTriangles


    def draw2D(self, cameraView, cameraLocation):
        img = np.zeros((400,400,3), np.uint8)
        img[:] = (255,255,255)
        minX, minY, maxX, maxY = cameraLocation[0], cameraLocation[1], cameraLocation[0], cameraLocation[1]
        for obj in self.entities:
            minX = min(np.min(obj.pts[0]), minX)
            maxX = max(np.max(obj.pts[0]), maxX)

            minY = min(np.min(obj.pts[1]), minY)
            maxY = max(np.max(obj.pts[1]), maxY)


        # print("max", maxX,minX, maxY, minY)
        m = max(maxX-minX, maxY-minY)
        if m == 0:
            print("no entities to map")
            return
        scale0 = 400/m
        minX -= m*0.1
        minY -= m*0.1
        maxX += m*0.1
        maxY += m*0.1
        scale = min(400/max(maxX-minX, maxY-minY), 400)
        # scale = 40

        for x in range(int(max(maxX-minX, maxY-minY))+1):
            # x=0
            buffer = (int(minX)-minX)
            cv2.line(img, (int((x+buffer)*scale), 0), ((int((x+buffer)*scale)), img.shape[0]), (0,0,0), 1)

        for y in range(int(max(maxX-minX, maxY-minY))+1):
            # y=0
            buffer = (int(minY)-minY)
            cv2.line(img, (0, int((y+buffer)*scale)), (img.shape[1], (int((y+buffer)*scale))), (0,0,0), 1)

        for obj in self.entities:
            for i, tri in enumerate(obj.triangles):
                pts = [[int((obj.pts[0][t]-minX)*scale), img.shape[0]-int((obj.pts[1][t]-minY)*scale)] for t in tri]
                cv2.fillPoly(img, np.array([pts]), obj.triangle_colors[i])

        cv2.rectangle(img, (int((cameraLocation[0]-minX)*scale), img.shape[0]-int((cameraLocation[1]-minY)*scale)), (int((cameraLocation[0]-minX)*scale), img.shape[0]-int((cameraLocation[1]-minY)*scale)), (255,0,0), 10)
        cv2.arrowedLine(img, (int((cameraLocation[0]-minX)*scale), img.shape[0]-int((cameraLocation[1]-minY)*scale)),    
                        (int((cameraLocation[0]-minX)*scale+math.cos(-cameraView[0]-math.pi/2)*40), img.shape[0]-int((cameraLocation[1]-minY)*scale+math.sin(-cameraView[0]-math.pi/2)*40)), (255,0,0), 2)

        cv2.imshow("2D", img)
        





    def draw3D(self, cameraView, cameraLocation, mask=False):
        """
        projects 3d objects from a perspective based on a given camera position and direction.
        Inputs: cameraview: [x,y,z]
               cameraLocation: [yaw, pitch] (radians)

        """

        fov_range = np.pi/3
        cameraView[0] %= np.pi*2
        cameraView[1] %= np.pi*2
        cameraLocation = np.array(cameraLocation)
        cameraView = np.array(cameraView)


        D = np.array([
            np.cos(cameraView[1]) * np.cos(np.pi*3/2-cameraView[0]),
            np.cos(cameraView[1]) * np.sin(np.pi*3/2-cameraView[0]),
            np.sin(cameraView[1])
        ])


        color_img = np.zeros((self.ht, self.wt, 3), np.uint8)

        if not mask:
            color_img[:] = (255, 180, 180)
            groundColor = (0,140,50)#(15, 60, 120)
            horiz = int(math.sin(cameraView[1])/math.cos(cameraView[1])*self.wt/2+self.ht/2);
            if (horiz>0):
                cv2.rectangle(color_img, (0, horiz), (self.wt, self.ht), groundColor, -1);
            else:
                cv2.rectangle(color_img, (0, 0), (self.wt, self.ht), groundColor, -1);


        s1 = math.sin(cameraView[0])
        c1 = math.cos(cameraView[0])
        s2 = math.sin(cameraView[1])
        c2 = math.cos(cameraView[1])

        objsToUse = [];
        t = time.time()
        for i in self.entities: # rotate the object's limits and check if they are in sight of the camera

            AB = i.limits[:, 0] - cameraLocation
            angle_diff0 = np.arccos(np.dot(AB, D) / (np.linalg.norm(AB) * np.linalg.norm(D)))

            AB = i.limits[:, 0] - cameraLocation
            angle_diff1 = np.arccos(np.dot(AB, D) / (np.linalg.norm(AB) * np.linalg.norm(D)))
            
            if angle_diff0 > fov_range and angle_diff1 > fov_range:
                i.triangle_colors = [(0,0,255)]*len(i.triangles)
            else:
                d = (cameraLocation[0]-i.limits[0][0])**2 + (cameraLocation[1]-i.limits[1][0])**2 + (cameraLocation[2]-i.limits[2][0])**2 # squared distance from camera
                if d < 90000:
                    objsToUse.append([-d, i, d])
                else:
                    print("too far")

                i.triangle_colors = [(0,255,0)] * len(i.triangles)


        objsToUse.sort(key = lambda x: x[0])
        # print("a", t-time.time())

        t = time.time()
        for otu in objsToUse:

            triangles = otu[1].triangles
            pts = otu[1].pts
            
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

            pi = np.where(np.max(triZ, axis=1) < 0)[0]
            # print(pi, triX, triY)
            polys = np.array((triX[pi], triY[pi]))


            c, n = polys.shape[1], polys.shape[2]
            polygons_list = [polys[:, i, :].T.reshape((-1, 1, 2)) for i in range(c)]

            # Draw each polygon
            for ind, poly in enumerate(polygons_list):
                if np.max(np.abs(poly)) < 1000:
                    if mask:
                        cv2.fillPoly(color_img, [poly.astype(np.int32)], (255,255,255))
                    else:
                        cv2.fillPoly(color_img, [poly.astype(np.int32)], otu[1].color)
                    # cv2.polylines(color_img, [poly.astype(np.int32)], True, (0,0,0), 1)
                        
        # print("b", t-time.time())


        


        return color_img



if __name__ == "__main__":
    cam = Camera(free_move=True)


    # for x in range(10):
    #     x*=10
    #     # x, y, = -10, 1
    #     for y in range(10):
    #         y*=10
    #         l, w = 1, 1
    #         angle = 0
    #         post, postTriangles = cam.getCube(x, y, l, w, angle, 0, 5)    
    #         cam.entities.append(Entity(post, postTriangles, (100,100,100)))
        

    x, y, = 10, 1
    l, w = 0.05, 0.05
    angle = 0
    post, postTriangles = cam.getCube(x-1, y, l, w, angle, 0, 5)    
    cam.entities.append(Entity(post, postTriangles, (100,0,100)))

    post, postTriangles = cam.getCube(x-1, y+6, l, w, angle, 0, 5)    
    cam.entities.append(Entity(post, postTriangles, (100,0,100)))


    # for i in range(3):
    x, y, = 10, 1
    makeGuardrail(cam, x, y, x, y+6, triangleCount=0)


    # corn = []
    # rows = 10
    # plants = 51
    # cornXY = []
    # x = 1
    # y= 0
    # for i in range(rows):
    #     for j in range(plants):
    #         xx = x + i / 3.281 / 12 * 30 # 30 inch rows
    #         yy = y + j / 3.281 / 12 * 4 # 4 inch spacing
    #         lon, lat = y, x
    #         cornXY.append([xx, yy])

    # for c in cornXY:
    #     c[0]*=30
    #     c[1]*=30
    #     cam.makeCorn(c[0], c[1], 1.3, 60, 3, 0)


    cameraView = [4.6,0]
    cameraLocation = [5,5,0.5]

    while True:
        cam.draw2D(cameraLocation=cameraLocation[:], cameraView=cameraView[:])
        color_img = cam.draw3D(cameraLocation=cameraLocation[:], cameraView=cameraView[:])
        cv2.imshow("3D", color_img)
        key = cv2.waitKey(0)
        speed = 0.05
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
            cameraLocation[2] += speed
        elif key == 225: # space
            cameraLocation[2] -= speed

        elif key == 97: # a
            cameraView[0] -= 0.1
        elif key == 100: # d
            cameraView[0] += 0.1

        elif key == 119: # w
            cameraView[1] = min(cameraView[1] + 0.1, math.pi*0.499)

        elif key == 115: # s
            
            cameraView[1] = max(cameraView[1] - 0.1, -math.pi*0.499)
        elif key == 27:
            exit()

        cameraView[0] = cameraView[0]%(2*math.pi)
