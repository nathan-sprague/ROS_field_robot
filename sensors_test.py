import time

testType = "GPS"

if testType == "serial":
    import serial

    arduino = serial.Serial(port='/dev/cu.SLAB_USBtoUART', baudrate=115200, timeout=.1)
    arduino.write(bytes("x", 'utf-8'))


elif testType == "GPS":
    from geographiclib.geodesic import Geodesic
    from gps import *

    gpsd = gps(mode=WATCH_ENABLE | WATCH_NEWSTYLE)
    while True:

        report = gpsd.next()  #
        if report['class'] == 'TPV':
            lat = getattr(report, 'lat', 0.0)
            lon = getattr(report, 'lon', 0.0)


elif testType == "Compass":
    from i2clibraries import i2c_hmc5883l

    hmc5883l = i2c_hmc5883l.i2c_hmc5883l(1)

    hmc5883l.setContinuousMode()

    # To scaled axes
    while True:
        (x, y, z) = hmc5883l.getAxes()
        print(x, y, z)
        time.sleep(0.5)
