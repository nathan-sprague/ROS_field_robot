import socket

UDP_IP = "192.168.1.1"
UDP_PORT = 5000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))


class SbfParser:
    HEADER_SIZE = 12  # Size of the SBF header in bytes

    def __init__(self, file_path):
        self.file_path = file_path
        self.file = open(file_path, 'rb')
        self.file.seek(0, 2)  # Seek to the end of the file
        self.file_size = self.file.tell()  # Get the file size
        self.file.seek(0)  # Reset the file pointer to the beginning

    def parse(self):
        while self.file.tell() < self.file_size:
            if self.file_size - self.file.tell() < self.HEADER_SIZE:
                break  # Insufficient data for a complete SBF message

            header = self.file.read(self.HEADER_SIZE)
            magic, length = struct.unpack('<4sI', header)
            if magic != b'SBF0':
                continue  # Not a valid SBF message

            if self.file_size - self.file.tell() < length:
                break  # Insufficient data for the complete message payload

            payload = self.file.read(length)
            # Process the payload (you can implement your own logic here)
            # For example, you can unpack specific data fields using struct

            # Print the payload as hex for demonstration purposes
            print(payload.hex())

            # Alternatively, you can parse the payload using struct
            # Example: assuming the payload contains two unsigned integers
            # data = struct.unpack('<II', payload)
            # print(data)

    def close(self):
        self.file.close()


while True:
    data, addr = sock.recvfrom(1024)
    sbf_data = pysbf.Sbf(data)
    print(sbf_data)

import struct



