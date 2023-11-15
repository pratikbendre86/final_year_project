import socket
import sys
import codecs
HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432        # The port used by the server
result = sys.argv[1]
#result = "1c:5f:2b:da:78:ec*-36*a6:ae:12:0e:37:ff*-38*1e:96:e6:3d:e2:df*-42*1e:4d:70:af:f8:9d*-71*"
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(bytes(result,'utf-8'))
    data = s.recv(1024)

print(codecs.decode(data, 'UTF-8'))