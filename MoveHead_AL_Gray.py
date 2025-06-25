import cv2
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
from datetime import timedelta
import time
import matplotlib.pyplot as plt

# ============ 参数设置 ============
# VIDEO_PATH = "recording_file/AL_FH.avi"
# SPO2_CSV = "recording_file/AL_FH_PRbmp_log.csv"
VIDEO_PATH = "recording_file/AL_MV.avi"
SPO2_CSV = "recording_file/AL_MV_PRbmp_log.csv"
METHOD = 'green'

LOWCUT = 0.65
HIGHCUT = 3.9
FILTER_ORDER = 3
SMOOTH_ALPHA = 0.2
MAX_HR_CHANGE = 10
MEDIAN_WINDOW = 5
MIN_HR = 40
MAX_HR = 160
SIGNAL_STD_THRESHOLD = 1.5
MIN_SIGNAL_LENGTH = 30

# ============ 数据 ============
spo2_df = pd.read_csv(SPO2_CSV)
spo2_df['Timestamp'] = pd.to_datetime(spo2_df['Timestamp'])

# ============ 函数 ============
def bandpass_filter(signal, fs=30, lowcut=LOWCUT, highcut=HIGHCUT, order=FILTER_ORDER):
    if len(signal) < 10:
        return signal
    nyq = 0.5 * fs
    low = max(0.01, lowcut / nyq)
    high = min(0.99, highcut / nyq)
    if low >= high:
        return signal
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def robust_estimate_hr_fft(signal, fs):
    if len(signal) < MIN_SIGNAL_LENGTH:
        return 0, 0.0
    freqs = np.fft.rfftfreq(len(signal), d=1/fs)
    fft_vals = np.abs(np.fft.rfft(signal))
    valid = (freqs * 60 > MIN_HR) & (freqs * 60 < MAX_HR)
    freqs = freqs[valid]
    fft_vals = fft_vals[valid]
    if len(fft_vals) == 0:
        return 0, 0.0
    max_idx = np.argmax(fft_vals)
    hr = freqs[max_idx] * 60
    quality = min(1.0, (fft_vals[max_idx] / np.sum(fft_vals)) * 3)
    return hr, quality

def get_spo2_hr(ts):
    diff = abs(spo2_df['Timestamp'] - ts)
    idx = diff.idxmin()
    return spo2_df.loc[idx, 'Pulse Rate (bpm)']

class HeartRateSmoother:
    def __init__(self):
        self.alpha = SMOOTH_ALPHA
        self.median_window = MEDIAN_WINDOW
        self.max_change = MAX_HR_CHANGE
        self.ema_hr = 70
        self.median_history = []
        self.last_valid = 70

    def update(self, new_hr, quality):
        if quality < 0.3 or new_hr < MIN_HR or new_hr > MAX_HR:
            return self.last_valid
        if abs(new_hr - self.last_valid) > self.max_change:
            new_hr = (new_hr + 2 * self.last_valid) / 3
        self.ema_hr = self.alpha * new_hr + (1 - self.alpha) * self.ema_hr
        self.median_history.append(self.ema_hr)
        if len(self.median_history) > self.median_window:
            self.median_history.pop(0)
        smoothed = np.median(self.median_history)
        self.last_valid = smoothed
        return smoothed

# ============ 主程序 ============
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("视频打开失败")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("人脸检测器加载失败")
        return

    signals = {'forehead': []}
    smoother = HeartRateSmoother()
    base_ts = spo2_df['Timestamp'].iloc[0]
    frame_count = 0
    last_face = None
    results = []

    # 初始化动态折线图
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    line1, = ax.plot([], [], 'b-', label='rPPG HR')
    line2, = ax.plot([], [], 'r-', label='SPO2 HR')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('HR (bpm)')
    ax.set_xlim(0, 60)
    ax.set_ylim(40, 120)
    ax.legend()
    ax.grid(True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        elapsed_sec = frame_count / fps
        video_ts = base_ts + timedelta(seconds=elapsed_sec)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(gray, 1.05, 2, minSize=(80, 80))
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            last_face = (x, y, w, h)
        elif last_face:
            (x, y, w, h) = last_face
        else:
            cv2.imshow("rPPG", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        fh_x = x + int(0.25 * w)
        fh_y = y + int(0.05 * h)
        fh_w = int(0.5 * w)
        fh_h = int(0.15 * h)
        forehead_roi = frame[fh_y:fh_y+fh_h, fh_x:fh_x+fh_w]
        signals['forehead'].append(np.mean(forehead_roi[:, :, 1]))

        if len(signals['forehead']) >= int(fps * 10):
            window_len = int(fps * 10)
            signal = bandpass_filter(signals['forehead'][-window_len:], fps)
            hr, quality = robust_estimate_hr_fft(signal, fps)
            spo2_hr = get_spo2_hr(video_ts)
            smoothed_hr = smoother.update(hr, quality)

            text1 = f"rPPG HR: {smoothed_hr:.1f} bpm"
            text2 = f"Spo2 HR: {spo2_hr:.1f} bpm"

            margin_right = 10

            (text_width1, text_height1), _ = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            (text_width2, text_height2), _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

            x1 = frame.shape[1] - text_width1 - margin_right
            x2 = frame.shape[1] - text_width2 - margin_right

            y1 = 30
            y2 = 60

            cv2.putText(frame, text1, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, text2, (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


            results.append({'time': elapsed_sec, 'rppg_hr': smoothed_hr, 'spo2_hr': spo2_hr})

            # 实时更新折线图
            times = [r['time'] for r in results]
            rppg_hrs = [r['rppg_hr'] for r in results]
            spo2_hrs = [r['spo2_hr'] for r in results]

            line1.set_data(times, rppg_hrs)
            line2.set_data(times, spo2_hrs)
            ax.set_xlim(max(0, times[-1] - 60), times[-1] + 5)
            ax.relim()
            ax.autoscale_view(scaley=True)
            fig.canvas.draw()
            fig.canvas.flush_events()

        cv2.imshow("rPPG", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    # plt.ioff()

    if results:
        df = pd.DataFrame(results)
        df.to_csv('hr_results.csv', index=False)
        print("已保存心率数据到 hr_results.csv")

if __name__ == "__main__":
    main()
