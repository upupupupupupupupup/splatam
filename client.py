import time

from multiprocessing.connection import Client

client = Client(('127.0.0.1', 8000))

while True:
    data = client.recv()  # 等待接受数据
    print(data)
    data = 'get data'
    client.send(data)
    data = client.recv()  # 等待接受数据
    time.sleep(1)
