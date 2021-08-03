import math

def findShortestAngle(targetHeading, heading):
    # finds the smaller angle between one heading or another (go negative or positive)
    
    steerDif = targetHeading - heading
    if steerDif > 180:
        steerDif = steerDif - 360
    elif steerDif < -180:
        steerDif = steerDif + 360
    return steerDif

def findSteerAngle(targetHeading, heading):
    # limit angle to less than 45 degrees either way

    steerDif = findShortestAngle(targetHeading, heading)
    if steerDif > 45:
        steerDif = 45
    elif steerDif < -45:
        steerDif = -45
    return steerDif


def findDistBetween(coords1, coords2):
    # finds distance between two coordinates. Corrects for longitude

    # 1 deg lat = 364,000 feet
    # 1 deg long = 288,200 feet
    x = (coords1[1] - coords2[1]) * 364000

    longCorrection = math.cos(coords1[0] * math.pi / 180)
    y = (coords1[0] - coords2[0]) * longCorrection * 364000

    return x, y

def atDestination(coords1, coords2, tolerance=5.0):
    # checks if the two points are on top of each other within specified tolerance

    x, y = findDistBetween(coords1, coords2)
    # print("distance away: ", x, y)
    if x * x + y * y < tolerance * tolerance:
        return True
    else:
        return False


def findAngleBetween(coords1, coords2):
    # finds angle between two points, with zero being north. 
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

    return realAngle






def makePath(currentCoords, destination, destHeading, turnRadius):
    # This function finds the best sub-destination for the robot so it can reach the destination at the desired heading.
    # The robot will go to the sub-destination and will go to the next calculated one until it reaches the real destination.

    longCorrection = math.cos(currentCoords[0] * math.pi / 180)

    offsetY = turnRadius / 12 / 364000 * math.sin((90 + destHeading) * math.pi / 180)
    offsetX = turnRadius / 12 / 364000 * math.cos((90 + destHeading) * math.pi / 180) * longCorrection
    approachCircleCenter1 = [destination[0] + offsetX, destination[1] + offsetY]

    offsetY = turnRadius / 12 / 364000 * math.sin((-90 + destHeading) * math.pi / 180)
    offsetX = turnRadius / 12 / 364000 * math.cos((-90 + destHeading) * math.pi / 180) * longCorrection
    approachCircleCenter2 = [destination[0] + offsetX, destination[1] + offsetY]

    x1, y1 = findDistBetween(currentCoords, approachCircleCenter1)
    x2, y2 = findDistBetween(currentCoords, approachCircleCenter2)

    dist1 = x1 * x1 + y1 * y1
    dist2 = x2 * x2 + y2 * y2
    print(dist1, dist2)

    if dist1 < dist2:
        print("clockwise approach")
        clockwise = True
        closerApproach = approachCircleCenter1
    else:
        print("Counter clockwise approach")
        clockwise = False
        closerApproach = approachCircleCenter2

    print(currentCoords, closerApproach)

    a = findAngleBetween(currentCoords, closerApproach) * 180 / math.pi
    print("angle", a)

    subPoints = []
    if clockwise:
        offsetY = turnRadius / 12 / 364000 * math.sin((a - 90) * math.pi / 180)
        offsetX = turnRadius / 12 / 364000 * math.cos((a - 90) * math.pi / 180) * longCorrection
        approachPoint1 = [closerApproach[0] + offsetX, closerApproach[1] + offsetY]
    
        b = findAngleBetween(currentCoords, approachPoint1) * 180 / math.pi
        subPoints = [approachPoint1]

    else:
        offsetY = turnRadius / 12 / 364000 * math.sin((a + 90) * math.pi / 180)
        offsetX = turnRadius / 12 / 364000 * math.cos((a + 90) * math.pi / 180) * longCorrection
        approachPoint2 = [closerApproach[0] + offsetX, closerApproach[1] + offsetY]

        subPoints = [approachPoint2]
        c = findAngleBetween(currentCoords, approachPoint2) * 180 / math.pi

    return subPoints


