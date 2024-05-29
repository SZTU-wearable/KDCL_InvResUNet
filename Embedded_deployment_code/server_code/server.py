# -*- coding: utf-8 -*-

# server.py
import socket
import numpy as np
import onnxruntime

session_options = onnxruntime.SessionOptions()
session_options.intra_op_num_threads = 4  # 设置内部操作的线程数
session_options.inter_op_num_threads = 2  # 设置操作间的线程数

# 加载 ONNX 模型
onnx_model = onnxruntime.InferenceSession("model_test01.onnx", session_options)

HOST = '10.102.6.211'  # 服务器的IP地址
PORT = 12346           # 服务器监听的端口

# 创建socket对象
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 绑定IP地址和端口号
server_socket.bind((HOST, PORT))

# 开始监听
server_socket.listen(1)
print("Server listening for connections...")

conn, addr = server_socket.accept()
print(f"Connected by {addr}")

def receive_array(conn, shape, dtype):
    # 根据形状和数据类型计算数据字节大小
    data_length = np.prod(shape) * np.dtype(dtype).itemsize
    data = conn.recv(data_length)
    while len(data) < data_length:
        more_data = conn.recv(data_length - len(data))
        if not more_data:
            raise Exception("Short read from socket")
        data += more_data
    return np.frombuffer(data, dtype=dtype).reshape(shape)

try:
    while True:
        # 假设我们知道x和y的形状和数据类型
        x_shape = (4, 625)
        y_shape = (1, 625)
        dtype = 'float32'  # 假设发送的数据类型为float32
        
        # 接收x数组
        x_array = receive_array(conn, x_shape, dtype)
        # print("Received x:", x_array.shape)

        # 接收y数组
        y_array = receive_array(conn, y_shape, dtype)
        # print("Received y:", y_array.shape)
        x_array = np.expand_dims(x_array, axis=0)
        output = onnx_model.run(None, {"input": x_array})
        output = np.array(output)
        output = output.flatten()
        conn.send(output.tobytes())

except Exception as e:
    print("Error:", e)
finally:
    conn.close()


