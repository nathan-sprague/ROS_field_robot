import serial
a = serial.Serial(port='/dev/ttyUSB0', baudrate=115200, timeout=.1)

b = serial.Serial(port='/dev/ttyUSB1', baudrate=115200, timeout=.1)
   

a.write(bytes("f0", 'utf-8'))


a.write(bytes("p0", 'utf-8'))

b.write(bytes("f0", 'utf-8'))

b.write(bytes("p0", 'utf-8'))
   
