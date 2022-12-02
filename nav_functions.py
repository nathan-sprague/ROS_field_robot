import math
import numpy as np
import pyproj
from shapely.geometry import Point, Polygon

"""
This file contains a bunch of helpful files functions used to help navigate.
This file does nothing on its own.

"""




def findShortestAngle(targetHeading, heading):
    # finds the smaller angle between one heading or another (go negative or positive)
    # uses degrees
    
    steerDif = targetHeading % 360 - heading % 360

    if steerDif > 180:
        steerDif = steerDif - 360
    elif steerDif < -180:
        steerDif = steerDif + 360

    return steerDif

# print(findShortestAngle(355, 360*3+100))

# def findSteerAngle(targetHeading, heading):
#     # limit angle to less than 45 degrees either way

#     steerDif = findShortestAngle(targetHeading, heading)
#     if steerDif > 45:
#         steerDif = 45
#     elif steerDif < -45:
#         steerDif = -45
#     return steerDif


def findDistBetween0(coords1, coords2):
    # finds distance between two coordinates in feet. Corrects for longitude

    # 1 deg lat = 364,000 feet
    # 1 deg long = 288,200 feet
    # print("dx", coords1[1]-coords2[1])
    x = (coords1[1] - coords2[1]) * 364000

    longCorrection = math.cos(coords1[0] * math.pi / 180)
    y = (coords1[0] - coords2[0]) * longCorrection * 364000
    # print("dy", coords1[0]-coords2[0])

    return [x, y]
    

def findDistBetween(coords1, coords2):
    # print("fdb0", findDistBetween0(coords1, coords2))
    p = pyproj.Proj('epsg:2793')
    # p = pyproj.Proj(proj='utm',zone=16,ellps='WGS84', preserve_units=False)

    x,y = p(coords1[1], coords1[0])

    x2,y2 = p(coords2[1], coords2[0])

    a = [(x-x2)*3.28084, (y-y2)*3.28084]
    # print("a", a)
    # print("new", (a[0]**2 + a[1]**2)**0.5)
    # print("fdb1", a,"\n")
    return a


def findDistBetweenAlt2(coords1, coords2):
    lat1 = coords1[0]
    lat2 = coords2[0]
    long1 = coords1[1]
    long2 = coords2[1]

    lat1 *=math.pi/180;
    lat2 *=math.pi/180;
    long1*=math.pi/180;
    long2*=math.pi/180;

    dlong = (long2 - long1);
    dlat  = (lat2 - lat1);

    # Haversine formula:
    R = 6371;
    a = math.sin(dlat/2)*math.sin(dlat/2) + math.cos(lat1)*math.cos(lat2)*math.sin(dlong/2)*math.sin(dlong/2)
    c = 2 * math.atan2( math.sqrt(a), math.sqrt(1-a) );
    d = R * c

    return a * 1000 * 3.28084


def findDistBetweenAlt(coords1, coords2):
    # returns a single number in ft
    lat1 = coords1[0]
    lat2 = coords2[0]
    lon1 = coords1[1]
    lon2 = coords2[1]
    p = math.pi / 180
    a = 0.5 - math.cos((lat2 - lat1) * p)/2 +  math.cos(lat1 * p) * math.cos(lat2 * p) * (1 - math.cos((lon2 - lon1) * p))/2
    return 12742 * math.asin(math.sqrt(a)) * 1000 * 3.28084; # 2 * R; R = 6371 km


# c1 = [40.4702929, -86.9949859]
# c2 = [c1[0]+0.00000003, c1[1]+0.03] #[40.4702924, -86.9949514]
# print(c2)


# a = findDistBetween(c1, c2)

# print("a", a)
# print("og", (a[0]**2 + a[1]**2)**0.5)

# print("alt1", findDistBetween(c1, c2))
# print("alt2", findDistBetween0(c1, c2))


def atDestination(coords1, coords2, tolerance=5.0):
    # checks if the two points are on top of each other within specified tolerance

    x, y = findDistBetween(coords1, coords2)

    if x * x + y * y < tolerance * tolerance:
        return True
    else:
        return False


def findAngleBetween(coords1, coords2):
    # finds angle between two points, with zero being north. Returns value in radians
    # This function could be substuted for some cross products and trig, but it works the same and I know how it works.

    x, y = findDistBetween(coords1, coords2)
    if x == 0:
        if coords1[1] > coords2[1]:  # directly above
            return math.pi
        else:  # directly below
            return 0

    slope = y / x
    angle = math.atan(slope)

    if coords1[1] < coords2[1]:
        realAngle = angle + math.pi * 1.5
    else:  # to the left
        realAngle = angle + math.pi * 0.5

    realAngle = 2 * math.pi - realAngle
    if realAngle > math.pi:
        realAngle = realAngle - 2 * math.pi

    if angle < 0:
        angle = 2*math.pi + angle

    return realAngle


def findAngleBetweenAlt(coords1, coords2):
    lat1 = coords1[0]
    lat2 = coords2[0]
    long1 = coords1[1]
    long2 = coords2[1]

    dLon = (long2 - long1);

    y = math.sin(dLon) * math.cos(lat2);
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1)* math.cos(lat2) * math.cos(dLon);

    brng = math.atan2(y, x);

    # brng *= 180 / math.pi
    brng = (brng + 2*math.pi) %  2*math.pi
    brng = 2*math.pi - brng # count degrees counter-clockwise - remove to make clockwise


    return brng


# c1 = [40.4702929, -86.9949559]
# c2 = [c1[0]+0.03, c1[1]+0.03]

# print("og", findAngleBetweenAlt(c1, c2)*180/math.pi)

# print("new", findAngleBetween(c1, c2)*180/math.pi )


def estimateCoords(robotCoords, robotHeading, robotSpeed, updateSpeed, proj=False):
    if proj == False:
        proj = pyproj.Proj('epsg:2793')

    turningSpeedConst = 3.7 / 0.3 * updateSpeed 
    movementSpeedConst = 0.35
    
    realHeadingChange = (robotSpeed[0]- robotSpeed[1])*turningSpeedConst
    robotHeading += realHeadingChange

    distMoved = (robotSpeed[0] + robotSpeed[1]) * 5280/3600/3.28 * movementSpeedConst
 

    x, y = proj(robotCoords[1], robotCoords[0])
    dy = distMoved * math.cos(robotHeading*math.pi/180) * updateSpeed
    dx = distMoved * math.sin(robotHeading*math.pi/180) * updateSpeed
    x += dx
    y += dy

    y2,x2 = proj(x, y, inverse=True)

    newCoords = [x2, y2]

    return newCoords, robotHeading

def polarToCoords(coords, heading, distance, proj=False):
    if proj == False:
        proj = pyproj.Proj('epsg:2793') 

    x, y = proj(coords[1], coords[0])
    dy = distance * math.cos(heading*math.pi/180)
    dx = distance * math.sin(heading*math.pi/180)
    x += dx
    y += dy

    y2,x2 = proj(x, y, inverse=True)

    newCoords = [x2, y2]

    return newCoords






def findDiffSpeeds(currentCoords, targetCoords, currentHeading, targetHeading, finalHeading = False, turnConstant = 0, destTolerance = 5, obstacles=[]):
    ### NOTE: RIGHT NOW IT DOESNT USE ALL THE PARAMETERS. WILL EVENTUALLY MAKE IT "SMARTER"
    """
    Finds the optimal speed for each wheel to reach the destination, at the desired heading. 
    This allows the robot to turn as it is moving toward its target, rather than doing a 0-point turn every time
    The further the robot is from the target, the robot will make a larger radius turn.

    parameters:
    distToTarget: distance robot is from target (tuple)
    currentHeading: heading of robot (degrees)
    targetHeading: direction of target (degrees)
    finalHeading: angle the robot should end at, if none then False (degrees or False)
    turnConstant: How sharp the turns should be. If the TC is 0, the turn sharpness will vary based on distance
    destTolerance: point at which the robot will begin pointing toward its final heading

    returns:
    wheel speed (list). Both values are between -100 and 100
    """

    # examples:
    # heading difference: 0; distToTarget: 10; --> [100, 100] (just go straight with both wheels)
    # heading difference: 180; distToTarget: 10; --> [100, -100] (turn around, zero point turn)
    # heading difference: 90; distToTarget: 0; --> [100, -100] (turn but dont move, zero point turn)
    # heading difference: -90; distToTarget: 0; --> [-100, 100] (same as above, but other direction)


    distToTarget = findDistBetween(currentCoords, targetCoords)


    # at destination, just do a 0 point turn to get at correct final heading
    if distToTarget[0]**2 + distToTarget[1]**2 < destTolerance ** 2:
        # print("\n\n\nat destination - from nav function")
        if finalHeading != False:
            return find0ptTurnSpeed(currentHeading, targetHeading, 5)



    dist1 = (distToTarget[0]*distToTarget[0] + distToTarget[1]*distToTarget[1])**0.5 # linear distance from target in feet


    headingDiff = findShortestAngle(targetHeading, currentHeading)

    print("targetHeading", targetHeading, "currentHeading", currentHeading, "headingDiff", headingDiff)


    if turnConstant == 0:

        hd = abs(headingDiff)
        if dist1 > 10:
            # turnConstant = 2
            if hd > 10:
                turnConstant = 2
            else:
                turnConstant = 0.5
              #  turnConstant = (10-hd)
        elif dist1 > 0:
            turnConstant = hd / dist1

        # print("tc", turnConstant)


    if abs(headingDiff) > 140 and distToTarget[0]**2 + distToTarget[1]**2 < 4**2: # target is close and the robot passed it. Just back up a bit
        return [-20, -20]


        

    headingDiff *= turnConstant


    # print("headingDiff", headingDiff / turnConstant, headingDiff)
    
    if abs(headingDiff) > 180: # target is behind you basically. you will do a 0-point turn no matter what
        headingDiff = 180 * headingDiff / abs(headingDiff)
        return find0ptTurnSpeed(currentHeading, targetHeading, 5)


    fasterSpeed = 100


    if dist1 < 5:
        fasterSpeed = 30

    slowerSpeed = fasterSpeed - (fasterSpeed*2) * abs(headingDiff/180)*1 # multiply by a number less than 1 to get a wider turn


    robotSpeed = [fasterSpeed, fasterSpeed]
    if headingDiff > 0:
        robotSpeed = [fasterSpeed, slowerSpeed]
    elif headingDiff < 0:
        robotSpeed = [slowerSpeed, fasterSpeed]


    if False: # len(obstacles) > 0:

        futureCoords, _ = estimateCoords(currentCoords, currentHeading, (robotSpeed[0]/100, robotSpeed[1]/100), 2)
        p1 = Point(futureCoords[0], futureCoords[1])
        obstructions = False

        for o in obstacles:
            poly = Polygon(o)

            if p1.within(poly):
                obstructions = True
                print("\n\n\n\nWILL HIT OBSTACLE!!!!!")
                break

        if obstructions:
            robotSpeed = [-50, 50]

    return robotSpeed

def pointInPoly(point, poly):
    poly = Polygon(poly)
    point = Point(point[0], point[1])
    if point.within(poly):
        return True
    return False




def find0ptTurnSpeed(heading, targetHeading, tolerance):
    # Slows down the turn rate as the robot gets closer to the target heading
    minSpeed = 10
    maxSpeed = 80

    headingDiff = findShortestAngle(targetHeading, heading)

    headingDiffMag = abs(headingDiff)# * 180/3.14

    turnSpeed = maxSpeed

    if headingDiffMag < 90:
        turnSpeed = minSpeed + (maxSpeed-minSpeed)*(headingDiffMag/90)/2


    if headingDiff < 0:
        return [-turnSpeed, turnSpeed]
    else:
        return [turnSpeed, -turnSpeed]

    return [minSpeed, minSpeed]


def makePath(currentCoords, currentHeading, targetCoords, finalHeading = False, obstacles = []):
    # This function estimates the path that the robot will take, assuming perfect GPS and heading accuracy
    # returns the path as a list of many tuple coordinates

    i = 0
    proj = pyproj.Proj('epsg:2793')
    subPoints = [currentCoords[:]]

    coords = currentCoords[:]
    while i<30 and (abs(coords[0] - targetCoords[0]) > 0.0000001 or abs(coords[1]-targetCoords[1]) > 0.0000001):

        targetHeading = math.degrees(findAngleBetween(coords, targetCoords))

        targetSpeedPercent = findDiffSpeeds(coords, targetCoords, currentHeading, targetHeading, finalHeading)

        robotSpeed = [targetSpeedPercent[0]/100, targetSpeedPercent[1]/100]

        coords, currentHeading = estimateCoords(coords, currentHeading, robotSpeed, 2, proj)
        if subPoints[-1] != coords:
            subPoints += [coords]
        i+=1
    print("\n\n\nsubPoints", subPoints)

    return subPoints

# print(makePath([40.4705552, -86.9952982], 40, [40.470558, -86.995371]))



# c1 = [40.4702929, -86.9949559]
# c2 = [c1[0]+0.03, c1[1]+0.03]
# print(c1, c2)

# geod = pyproj.Geod(ellps='WGS84')
# # print(geod.crs)


# lat0, lon0 = c1
# lat1, lon1 = c2

# azimuth1, azimuth2, distance = geod.inv(lon0, lat0, lon1, lat1)
# print("dist", distance)
# print('    azimuth', azimuth1, azimuth2)

#p = pyproj.Proj(proj='utm',zone=6, ellps='WGS84', preserve_units=False) #EPSG:32606 

