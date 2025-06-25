import cv2
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, detrend, welch
import time
import matplotlib.pyplot as plt

# ============ 参数设置 ============
VIDEO_PATH = "recording_file/CL_FH.avi"
SPO2_CSV = "recording_file/CL_FH_PRbmp_log.csv"
METHOD = 'GREEN'  # 'GREEN', 'CHROM', 'POS' 可选

# ============ 读取血氧计心率数据 ============
spo2_df = pd.read_csv(SPO2_CSV)
spo2_df['Timestamp'] = pd.to_datetime(spo2_df['Timestamp'])

def get_spo2_hr(ts):
    diff = abs(spo2_df['Timestamp'] - ts)
    idx = diff.idxmin()
    return spo2_df.loc[idx, 'Pulse Rate (bpm)']

# ============ 滤波器带通 ============
def bandpass_filter(signal, fs, lowcut=0.8, highcut=1.5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, signal)

# ============ 用Welch估计心率 ============
def estimate_hr_welch(signal, fs):
    f, pxx = welch(signal, fs, nperseg=min(len(signal), fs*10))
    mask = (f*60 >= 40) & (f*60 <= 120)
    if not np.any(mask):
        return 0
    peak_freq = f[mask][np.argmax(pxx[mask])]
    return peak_freq * 60

# ============ 初始化视频 ============
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # 备用帧率

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

signals = []
times = []

plt.ion()
fig, ax = plt.subplots()
line_rppg, = ax.plot([], [], label="rPPG HR")
line_spo2, = ax.plot([], [], label="SPO2 HR")
ax.set_ylim(40, 110)
ax.set_xlim(0, 60)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Heart Rate (bpm)")
ax.legend()

results = []

# 用于心率跳变限制的历史心率
hr_history = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    video_ts = spo2_df['Timestamp'].iloc[0] + pd.to_timedelta(ts, unit='s')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(100,100))

    if len(faces) > 0:
        x, y, w, h = faces[0]

        # 画脸框
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        roi = frame[y:y+h, x:x+w]
        roi = cv2.resize(roi, (100, 100))
        mean_rgb = np.mean(roi.reshape(-1, 3), axis=0)
        signals.append(mean_rgb)
        times.append(ts)

        # 只处理至少10秒的数据
        if len(signals) >= int(fps * 10):
            sig_array = np.array(signals[-int(fps*10):])

            if METHOD == 'GREEN':
                sig = sig_array[:,1]
            elif METHOD == 'CHROM':
                X = 3 * (sig_array[:,0] - sig_array[:,1])
                Y = 1.5 * (sig_array[:,0] + sig_array[:,1] - 2*sig_array[:,2])
                sig = X / (X.std()+1e-8) - Y / (Y.std()+1e-8)
            elif METHOD == 'POS':
                X = sig_array[:,0] - sig_array[:,1]
                Y = sig_array[:,0] + sig_array[:,1] - 2 * sig_array[:,2]
                sig = X + Y
            else:
                sig = sig_array[:,1]

            sig = detrend(sig)
            sig = bandpass_filter(sig, fps)

            hr_raw = estimate_hr_welch(sig, fps)
            # 限制心率范围40-100bpm
            hr_raw = max(40, min(100, hr_raw))

            # 跳变限制，最大允许跳变40 bpm
            if hr_history:
                last_hr = hr_history[-1]
                if abs(hr_raw - last_hr) > 40:
                    hr_raw = last_hr + 40 * np.sign(hr_raw - last_hr)

            hr_history.append(hr_raw)

            spo2_hr = get_spo2_hr(video_ts)

            results.append({'time': ts, 'rppg_hr': hr_raw, 'spo2_hr': spo2_hr})

            # 限制绘图x轴范围自动扩展
            if ts > ax.get_xlim()[1]:
                ax.set_xlim(0, ts+10)

            # 计算平滑rPPG心率，用过去3个点均值
            window_size = 3
            if len(results) >= window_size:
                hr_smooth = np.mean([r['rppg_hr'] for r in results[-window_size:]])
            else:
                hr_smooth = hr_raw

            cv2.putText(frame, f"rPPG HR: {hr_smooth:.1f} bpm", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            cv2.putText(frame, f"SPO2 HR: {spo2_hr:.1f} bpm", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            times_plot = [r['time'] for r in results]
            rppg_plot = [r['rppg_hr'] for r in results]
            spo2_plot = [r['spo2_hr'] for r in results]

            line_rppg.set_data(times_plot, rppg_plot)
            line_spo2.set_data(times_plot, spo2_plot)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
    else:
        # 无脸时清空缓存，防止跳变
        signals.clear()
        times.clear()
        hr_history.clear()

    cv2.imshow("rPPG Heart Rate Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
