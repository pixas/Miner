# utils/allocate_port.py
import socket
import sys 
import random 
min_port = int(sys.argv[1])
max_port = int(sys.argv[2])
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        success = False 
        while not success:
            port = random.randint(min_port, max_port)
        # for port in range(min_port, max_port):
            
            try:
                s.bind(('', port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise IOError("No free port found in range")
print(find_free_port())