import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import QThread, pyqtSignal
import pyqtgraph as pg
from config import HOST, PORT
import socket
import h5py
import time
import numpy as np
import ctypes
from scipy import signal

def resample_signal(input_signal, current_sample_rate, target_sample_rate):
    # 计算重采样比例
    resample_ratio = target_sample_rate / current_sample_rate
    # 计算目标采样点数量
    target_num_samples = int(len(input_signal) * resample_ratio)
    # 使用scipy的resample函数进行重采样
    resampled_signal = signal.resample(input_signal, target_num_samples)
    return resampled_signal

def receive_array(conn, shape, dtype):
    data_length = np.prod(shape) * np.dtype(dtype).itemsize
    data = conn.recv(data_length)
    while len(data) < data_length:
        more_data = conn.recv(data_length - len(data))
        if not more_data:
            raise Exception("Short read from socket")
        data += more_data
    return np.frombuffer(data, dtype=dtype).reshape(shape)

class DataThread(QThread):
    data_received = pyqtSignal(np.ndarray)  # 创建一个信号，用于发送数据
    def run(self):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((HOST, PORT))

        try:
            # 加载DLL
            dll_path = r'cpp\cmake-build-debug\libget_ecg_ppg_data.dll'
            lib = ctypes.CDLL(dll_path)
            # get_ecg_ppg_data函数原型
            lib.get_ecg_ppg_data.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
            lib.get_ecg_ppg_data.restype = ctypes.c_int
            lib.start_api()
            # 创建一个双精度浮点数数组
            sec = 10
            freq = 200
            id = 0
            while True:
                buff = np.zeros(sec * freq * 2, dtype=np.double)
                # 调用函数
                result = lib.get_ecg_ppg_data(buff.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), sec)
                ecg_data = buff[0::2]
                ppg_data = buff[1::2]
                ecg_data = resample_signal(ecg_data, 200, 62.5)
                ppg_data = resample_signal(ppg_data, 200, 62.5)
                ecg_data, ppg_data = np.array(ecg_data, dtype=np.float32), np.array(ppg_data, dtype=np.float32)
                input_data = np.vstack((ecg_data, ppg_data))
                client_socket.sendall(input_data.tobytes())
                print("id = ", id)
                pred_abp = receive_array(client_socket, (625,), 'float32')
                id += 1
                for i in range(len(ecg_data)):
                    plot_data = np.array([ecg_data[i], ppg_data[i], pred_abp[i]]).reshape((3, 1))
                    self.data_received.emit(plot_data)
                    self.msleep(10)
            lib.stop_api()
        finally:
            lib.stop_api()


class RealTimePlotWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Real-Time Waveforms with Scrolling Time Axis")
        # 设置窗口的主要布局
        layout = QVBoxLayout()
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # 初始化图表、曲线列表和数据存储
        self.plots = []
        self.curves = []
        self.data = []
        self.timeStamps = []

        # 创建三个图表：ECG、PPG和ABP
        # 定义每个图表的标题、颜色和字体大小
        # 创建三个图表：ECG、PPG和ABP
        # 定义每个图表的标题、颜色、字体大小和曲线粗细
        plot_details = [
            {"title": "ECG Waveform", "color": "#FF0000", "size": "14pt", "xlabel": "时间 (秒)", "ylabel": '<span style="font-size: 30px;">\n</span>', "penWidth": 3},  # 曲线粗细设置为3
            {"title": "PPG Waveform", "color": "#00FF00", "size": "14pt", "xlabel": "时间 (秒)", "ylabel": '<span style="font-size: 30px;">\n</span>',"penWidth": 3},  # 曲线粗细设置为2
            {"title": "ABP Waveform", "color": "#34E5EB", "size": "14pt", "xlabel": "时间 (秒)", "ylabel": '<span style="font-size: 30px;">mmHg</span>', "penWidth": 3}  # 曲线粗细设置为1
        ]

        for detail in plot_details:
            # 使用HTML格式设置标题，以包含字体大小和颜色
            title_html = f"<span style='font-size: {detail['size']}; color: {detail['color']};'>{detail['title']}</span>"
            plot = pg.PlotWidget(title=title_html)
            plot.setYRange(0, 1)
            # plot.setLabel('left', detail['ylabel'])
            plot.showGrid(x=False, y=False)
            # 设置X轴的标签
            plot.setLabel('bottom', detail['xlabel'])
            plot.setLabel('left', detail['ylabel'])

            curve = plot.plot(pen=pg.mkPen(color=detail['color'], width=detail['penWidth']))  # 设置曲线颜色和粗细
            layout.addWidget(plot)
            self.plots.append(plot)
            self.data.append(np.zeros(625))
            self.curves.append(curve)
            self.timeStamps.append(np.linspace(-5, 0, 625))

            # 根据图表类型设置Y轴范围
            if detail['title'].startswith("ABP"):
                plot.setYRange(40, 160)

            elif detail['title'].startswith("ECG"):
                plot.setYRange(-1, 1)
            elif detail['title'].startswith("PPG"):
                plot.setYRange(-5, 5)

        # 启动数据接收线程
        self.data_thread = DataThread()
        self.data_thread.data_received.connect(self.update)  # 将信号连接到更新函数
        self.data_thread.start()
    
    def update(self, in_data):
        # 更新数据和时间戳
        newTimeStamp = self.timeStamps[0][-1] + 0.016  # 基于最后一个时间戳增加
        for i in range(3):
            self.data[i] = np.roll(self.data[i], -1)  # 数据向左滚动
            self.data[i][-1] = in_data[i, 0]  # 添加新的随机数据点
            self.timeStamps[i] = np.roll(self.timeStamps[i], -1)
            self.timeStamps[i][-1] = newTimeStamp
        
        # 重绘图表
        for i, (curve, data, timeStamp) in enumerate(zip(self.curves, self.data, self.timeStamps)):
            curve.setData(timeStamp, data)

        # 更新横坐标范围，保持最近5秒数据可见
        for plot in self.plots:
            plot.setXRange(newTimeStamp - 10, newTimeStamp)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RealTimePlotWindow()
    window.show()
    window.resize(2000, 1200)  # 设置初始窗口大小
    sys.exit(app.exec_())


