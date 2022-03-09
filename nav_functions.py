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
    # finds distance between two coordinates in feet. Corrects for longitude

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





def findDiffWheelSpeeds(distToTarget, currentHeading, targetHeading, finalHeading = False, turnConstant = 10, destTolerance = 5):
    ### NOTE: RIGHT NOW IT JUST MAKES IT DO A 0-POINT TURN. IT DOES NOT DO WHAT IT SAYS IN THE DESCRIPTION


    """
    Finds the optimal speed for each wheel to reach the destination, at the desired heading. 
    This allows the robot to turn as it is moving toward its target, rather than doing a 0-point turn every time
    The further the robot is from the target, the robot will make a larger radius turn.

    parameters:
    distToTarget: distance robot is from target (feet)
    currentHeading: heading of robot (degrees)
    targetHeading: direction of target (degrees)
    finalHeading: angle the robot should end at (degrees)
    turnConstant: number based on the slip expected from the differential wheel speed
    destTolerance: point at which the robot will begin pointing toward its final heading

    returns:
    wheel speed (list)
    """

    # examples:
    # heading difference: 0; distToTarget: 10; --> [100, 100] (just go straight with both wheels)
    # heading difference: 180; distToTarget: 10; --> [100, -100] (turn around, zero point turn)
    # heading difference: 90; distToTarget: 0; --> [100, -100] (turn but dont move, zero point turn)
    # heading difference: -90; distToTarget: 0; --> [-100, 100] (same as above, but other direction)

    headingDiff = findSteerAngle(targetHeading, currentHeading)

    if headingDiff > 20:
        return [100, -100]

    elif headingDiff < 20:
        return [100, -100]

    if distToTarget < destTolerance and finalHeading != False:
        headingDiff = findSteerAngle(finalHeading, currentHeading)
        if headingDiff > 20:
           return [100, -100]

        elif headingDiff < 20:
            return [100, -100]
    
    return [100,100]





