import sys
import time
import numpy as np
import cv2
from PyQt5 import QtCore, QtWidgets, QtGui
from scipy.signal import butter, filtfilt
from cms50d import CMS50D
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ------------------ Bandpass Filter ------------------
def bandpass_filter(signal, fs=20, lowcut=0.7, highcut=3.5, order=2):
    if len(signal) < 5:
        return signal
    nyq = 0.5 * fs
    low = max(0.01, lowcut / nyq)
    high = min(0.99, highcut / nyq)
    if low >= high:
        return signal
    b, a = butter(order, [low, high], btype='band')
    try:
        return filtfilt(b, a, signal)
    except Exception as e:
        print(f"Filter error: {e}")
        return signal

# ------------------ Heart Rate Estimation ------------------
def estimate_hr_fft(signal, fs=20, min_hr=40, max_hr=180):
    n = len(signal)
    if n < 5:
        return 0
    freqs = np.fft.rfftfreq(n, d=1/fs)
    fft_vals = np.abs(np.fft.rfft(signal - np.mean(signal)))
    valid = (freqs*60 >= min_hr) & (freqs*60 <= max_hr)
    if not np.any(valid):
        return 0
    idx = np.argmax(fft_vals[valid])
    hr = freqs[valid][idx]*60
    return hr

# ------------------ Heart Rate Smoother ------------------
class HeartRateSmoother:
    def __init__(self, alpha=0.1, median_window=7, max_change=5):
        self.alpha = alpha
        self.max_change = max_change
        self.median_window = median_window
        self.ema_hr = 70.0
        self.median_history = []
        self.last_valid = 70.0
    
    def update(self, new_hr):
        # Return last valid if new_hr invalid
        if new_hr < 40 or new_hr > 180:
            return self.last_valid
        
        # Limit sudden changes
        if abs(new_hr - self.last_valid) > self.max_change:
            new_hr = (new_hr + 2 * self.last_valid) / 3
        
        # Exponential smoothing
        self.ema_hr = self.alpha * new_hr + (1 - self.alpha) * self.ema_hr
        
        # Median filter buffer
        self.median_history.append(self.ema_hr)
        if len(self.median_history) > self.median_window:
            self.median_history.pop(0)
        
        # Median filter output
        smoothed_hr = np.median(self.median_history)
        self.last_valid = smoothed_hr
        
        return smoothed_hr

# ------------------ CMS50D Serial Reader Thread ------------------
class CMS50DReader(QtCore.QThread):
    new_hr = QtCore.pyqtSignal(float)

    def __init__(self, port='COM9'):
        super().__init__()
        self.monitor = None
        self.running = False
        self.port = port

    def run(self):
        try:
            self.monitor = CMS50D(port=self.port)
            self.monitor.connect()
            self.monitor.start_live_acquisition()
            self.running = True
            while self.running:
                data = self.monitor.get_latest_data()
                if data and data['pulse_rate'] > 0:
                    self.new_hr.emit(float(data['pulse_rate']))
                time.sleep(0.1)
        except Exception as e:
            print(f"CMS50D error: {e}")

    def stop(self):
        self.running = False
        if self.monitor:
            self.monitor.disconnect()

# ------------------ Main Window ------------------
class RPPGWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time rPPG Heart Rate Estimation vs Pulse Oximeter")
        self.resize(1200, 800)

        # Main layout
        main_layout = QtWidgets.QHBoxLayout()
        
        # Left layout: video and HR display
        left_layout = QtWidgets.QVBoxLayout()
        
        self.video_label = QtWidgets.QLabel()
        self.video_label.setFixedSize(640, 480)
        left_layout.addWidget(self.video_label)
        
        # Algorithm selection
        algorithm_layout = QtWidgets.QHBoxLayout()
        algorithm_label = QtWidgets.QLabel("Algorithm:")
        algorithm_label.setStyleSheet("font-weight: bold;")
        self.algorithm_combo = QtWidgets.QComboBox()
        self.algorithm_combo.addItem("GREEN (Green Channel)", "green")
        self.algorithm_combo.addItem("FUSION (Multi-region Fusion)", "fusion")
        self.algorithm_combo.addItem("GRAY (Gray Channel)", "gray")
        self.algorithm_combo.currentIndexChanged.connect(self.on_algorithm_changed)
        algorithm_layout.addWidget(algorithm_label)
        algorithm_layout.addWidget(self.algorithm_combo)
        algorithm_layout.addStretch()
        left_layout.addLayout(algorithm_layout)
        
        # Heart rate display
        hr_layout = QtWidgets.QHBoxLayout()
        self.hr_rppg_label = QtWidgets.QLabel("rPPG HR: -- bpm")
        self.hr_rppg_label.setStyleSheet("font-size: 20px; font-weight: bold; color: blue;")
        self.hr_spo2_label = QtWidgets.QLabel("PPG HR: -- bpm")
        self.hr_spo2_label.setStyleSheet("font-size: 20px; font-weight: bold; color: red;")
        hr_layout.addWidget(self.hr_rppg_label)
        hr_layout.addWidget(self.hr_spo2_label)
        left_layout.addLayout(hr_layout)
        
        # Status label
        self.status_label = QtWidgets.QLabel("Status: Waiting for data...")
        self.status_label.setStyleSheet("font-size: 16px;")
        left_layout.addWidget(self.status_label)
        
        # Right layout: charts
        right_layout = QtWidgets.QVBoxLayout()
        
        # Create chart
        self.figure = Figure(figsize=(6, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Heart Rate (bpm)')
        self.ax.set_title('rPPG vs Pulse Oximeter Heart Rate')
        self.ax.grid(True)
        self.line_rppg, = self.ax.plot([], [], 'b-', label='rPPG HR')
        self.line_spo2, = self.ax.plot([], [], 'r-', label='Pulse Oximeter HR')
        self.ax.legend()
        self.ax.set_ylim(40, 120)
        
        right_layout.addWidget(self.canvas)
        
        # Add layouts to main layout
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

        # Camera init
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.show_error("Cannot open camera")
            return

        # DNN face detection model load
        model_dir = "models"
        prototxt_path = model_dir + "/deploy.prototxt"
        model_path = model_dir + "/res10_300x300_ssd_iter_140000.caffemodel"
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

        # rPPG signal buffers
        self.fs = 20  # sampling frequency approx
        self.window_seconds = 15  # time window length in seconds
        self.green_signals = {'forehead': [], 'left_cheek': [], 'right_cheek': []}
        self.gray_signals = {'forehead': [], 'left_cheek': [], 'right_cheek': []}
        self.time_stamps = []
        self.rppg_hr_values = []
        self.spo2_hr_values = []
        self.time_values = []
        self.current_algorithm = "green"  # default algorithm

        # HR smoother
        self.hr_smoother = HeartRateSmoother(alpha=0.1, median_window=5, max_change=5)

        # Pulse Oximeter HR
        self.spo2_hr = 0

        # Start CMS50D thread
        self.cms_reader = CMS50DReader(port='COM9')
        self.cms_reader.new_hr.connect(self.update_spo2_hr)
        self.cms_reader.start()

        # Timer for video frame processing
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(int(1000 / self.fs))

    def on_algorithm_changed(self):
        """Handle algorithm change"""
        self.current_algorithm = self.algorithm_combo.currentData()
        algorithm_name = self.algorithm_combo.currentText()
        self.status_label.setText(f"Status: Switched to {algorithm_name}")
        
        # Clear buffers
        self.green_signals = {'forehead': [], 'left_cheek': [], 'right_cheek': []}
        self.gray_signals = {'forehead': [], 'left_cheek': [], 'right_cheek': []}
        self.time_stamps = []
        self.rppg_hr_values = []
        self.time_values = []
        self.hr_smoother = HeartRateSmoother(alpha=0.1, median_window=5, max_change=5)

    def show_error(self, message):
        QtWidgets.QMessageBox.critical(self, "Error", message)
        self.close()

    def update_spo2_hr(self, hr):
        self.spo2_hr = hr
        if hr < 20:
            self.hr_spo2_label.setText("Pulse Oximeter HR: finger out")
        else:
            self.hr_spo2_label.setText(f"Pulse Oximeter HR: {hr:.1f} bpm")


    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.status_label.setText("Status: Cannot read camera frame")
            return

        frame = cv2.resize(frame, (640, 480))

        # DNN face detection
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                    (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        h, w = frame.shape[:2]
        faces = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                faces.append((x1, y1, x2 - x1, y2 - y1))

        if len(faces) > 0:
            # Pick largest face
            x, y, fw, fh = max(faces, key=lambda rect: rect[2] * rect[3])
            self.status_label.setText(f"Status: Face detected | Algorithm: {self.algorithm_combo.currentText()}")
            cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 255, 0), 2)

            # Forehead ROI
            fh_x = x + int(0.25 * fw)
            fh_y = y + int(0.05 * fh)
            fh_w = int(0.5 * fw)
            fh_h = int(0.15 * fh)
            forehead_roi = frame[fh_y:fh_y+fh_h, fh_x:fh_x+fh_w]
            if forehead_roi.size > 0:
                cv2.rectangle(frame, (fh_x, fh_y), (fh_x+fh_w, fh_y+fh_h), (255, 0, 0), 2)
                green_forehead = np.mean(forehead_roi[:, :, 1])
                gray_forehead = np.mean(cv2.cvtColor(forehead_roi, cv2.COLOR_BGR2GRAY))
            else:
                green_forehead = 0
                gray_forehead = 0

            # Left cheek ROI
            lc_x = x + int(0.2 * fw)
            lc_y = y + int(0.4 * fh)
            lc_w = int(0.15 * fw)
            lc_h = int(0.36 * fh)
            left_cheek_roi = frame[lc_y:lc_y+lc_h, lc_x:lc_x+lc_w]
            if left_cheek_roi.size > 0:
                cv2.rectangle(frame, (lc_x, lc_y), (lc_x+lc_w, lc_y+lc_h), (0, 255, 255), 2)
                green_left_cheek = np.mean(left_cheek_roi[:, :, 1])
                gray_left_cheek = np.mean(cv2.cvtColor(left_cheek_roi, cv2.COLOR_BGR2GRAY))
            else:
                green_left_cheek = 0
                gray_left_cheek = 0

            # Right cheek ROI
            rc_x = x + int(0.65 * fw)
            rc_y = y + int(0.4 * fh)
            rc_w = int(0.15 * fw)
            rc_h = int(0.36 * fh)
            right_cheek_roi = frame[rc_y:rc_y+rc_h, rc_x:rc_x+rc_w]
            if right_cheek_roi.size > 0:
                cv2.rectangle(frame, (rc_x, rc_y), (rc_x+rc_w, rc_y+rc_h), (0, 255, 255), 2)
                green_right_cheek = np.mean(right_cheek_roi[:, :, 1])
                gray_right_cheek = np.mean(cv2.cvtColor(right_cheek_roi, cv2.COLOR_BGR2GRAY))
            else:
                green_right_cheek = 0
                gray_right_cheek = 0

            # Store signals
            now = time.time()
            self.green_signals['forehead'].append(green_forehead)
            self.green_signals['left_cheek'].append(green_left_cheek)
            self.green_signals['right_cheek'].append(green_right_cheek)

            self.gray_signals['forehead'].append(gray_forehead)
            self.gray_signals['left_cheek'].append(gray_left_cheek)
            self.gray_signals['right_cheek'].append(gray_right_cheek)

            self.time_stamps.append(now)

            # Initialize start_time once
            if not hasattr(self, 'start_time') or self.start_time is None:
                self.start_time = self.time_stamps[0]

            # Keep recent data only (sliding window)
            while self.time_stamps and (self.time_stamps[-1] - self.time_stamps[0]) > self.window_seconds:
                self.time_stamps.pop(0)
                self.green_signals['forehead'].pop(0)
                self.green_signals['left_cheek'].pop(0)
                self.green_signals['right_cheek'].pop(0)
                self.gray_signals['forehead'].pop(0)
                self.gray_signals['left_cheek'].pop(0)
                self.gray_signals['right_cheek'].pop(0)

            # Estimate HR if enough data
            if len(self.green_signals['forehead']) >= self.fs * 5:
                if self.current_algorithm == "green":
                    signal = np.array(self.green_signals['forehead'])
                    self.process_signal(signal, "GREEN algorithm: Forehead green channel")
                elif self.current_algorithm == "fusion":
                    signal_forehead = np.array(self.green_signals['forehead'])
                    signal_left = np.array(self.green_signals['left_cheek'])
                    signal_right = np.array(self.green_signals['right_cheek'])
                    signal = 0.5 * signal_forehead + 0.25 * signal_left + 0.25 * signal_right
                    self.process_signal(signal, "FUSION algorithm: Multi-region green channel fusion")
                elif self.current_algorithm == "gray":
                    signal = np.array(self.gray_signals['forehead'])
                    self.process_signal(signal, "GRAY algorithm: Forehead gray channel")
        else:
            self.status_label.setText("Status: No face detected")
            self.green_signals = {'forehead': [], 'left_cheek': [], 'right_cheek': []}
            self.gray_signals = {'forehead': [], 'left_cheek': [], 'right_cheek': []}
            self.time_stamps = []
            self.start_time = None  # Reset start_time when no face

        # Show frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_img)
        self.video_label.setPixmap(pixmap)

    def process_signal(self, signal, algorithm_name):
        filtered = bandpass_filter(signal, fs=self.fs)
        hr = estimate_hr_fft(filtered, fs=self.fs)
        hr_smooth = self.hr_smoother.update(hr)
        self.hr_rppg_label.setText(f"rPPG HR: {hr_smooth:.1f} bpm")

        self.rppg_hr_values.append(hr_smooth)
        # self.time_values.append(time.time() - self.time_stamps[0])
        self.time_values.append(self.time_stamps[-1])  # 直接存最新时间戳


        # 同步填充spo2_hr_values，使长度和time_values保持一致
        # 如果当前spo2_hr无效，则用0填充
        spo2_val = self.spo2_hr if self.spo2_hr > 20 else 0
        while len(self.spo2_hr_values) < len(self.time_values):
            self.spo2_hr_values.append(spo2_val)

        self.update_chart()
    # 在类初始化时，time_values、rppg_hr_values、spo2_hr_values 初始化就不变
# 但这里的 update_chart 函数是重点

    def update_chart(self):
        if len(self.time_values) == 0 or len(self.rppg_hr_values) == 0:
            return

        window_width = 10  # 固定窗口10秒
        max_time = self.time_values[-1]  # 最新时间
        min_time = max_time - window_width
        self.ax.set_xlim(min_time, max_time)
        if min_time < 0:
            min_time = 0

        # 只保留窗口内的数据索引
        valid_indices = [i for i, t in enumerate(self.time_values) if t >= min_time]

        if not valid_indices:
            return

        times = [self.time_values[i] for i in valid_indices]
        rppg_hrs = [self.rppg_hr_values[i] for i in valid_indices]

        self.line_rppg.set_data(times, rppg_hrs)

        if len(self.spo2_hr_values) >= len(self.rppg_hr_values):
            spo2_hrs = [self.spo2_hr_values[i] for i in valid_indices]
            self.line_spo2.set_data(times, spo2_hrs)

        self.ax.set_xlim(min_time, min_time + window_width)  # 固定宽度显示窗口
        self.ax.set_ylim(40, 120)

        self.canvas.draw()




    def closeEvent(self, event):
        self.cap.release()
        self.cms_reader.stop()
        self.cms_reader.wait()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = RPPGWindow()
    window.show()
    sys.exit(app.exec())