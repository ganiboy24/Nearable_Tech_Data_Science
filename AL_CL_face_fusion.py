import cv2
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch, find_peaks
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
from datetime import timedelta

# ============ 参数设置 - 请根据需求调整这些值 ============
VIDEO_PATH = "recording_file/CL_FH.avi"
SPO2_CSV = "recording_file/CL_FH_PRbmp_log.csv"

# VIDEO_PATH = "recording_file/AL_FH.avi"
# SPO2_CSV = "recording_file/AL_FH_PRbmp_log.csv"
METHOD = 'green'  # 选择算法：'green', 'pca', 'fusion'

# ============ 低通滤波器参数 - 关键调整区域 ============
LOWCUT = 0.7    # [可调] 高通截止频率 (Hz)
HIGHCUT = 3.5   # [可调] 低通截止频率 (Hz) - 主要调整此参数控制平滑度
FILTER_ORDER = 2  # [可调] 滤波器阶数

# ============ 心率平滑参数 ============
SMOOTH_ALPHA = 0.1           # [可调] 指数平滑因子
MAX_HR_CHANGE = 5            # [可调] 最大允许心率变化 (bpm)
MEDIAN_WINDOW = 7            # [可调] 中值滤波窗口大小
MIN_HR = 40                  # 合理心率下限
MAX_HR = 180                 # 合理心率上限

# ============ 信号质量参数 ============
SIGNAL_STD_THRESHOLD = 1.5   # [可调] 信号质量阈值
MIN_SIGNAL_LENGTH = 30       # 信号处理最小长度（帧）

# ============ 读血氧计心率数据 ============
spo2_df = pd.read_csv(SPO2_CSV)
spo2_df['Timestamp'] = pd.to_datetime(spo2_df['Timestamp'])

# ============ 信号处理函数 ============
def bandpass_filter(signal, fs=30, lowcut=LOWCUT, highcut=HIGHCUT, order=FILTER_ORDER):
    if len(signal) < 10:
        return signal
    
    nyq = 0.5 * fs
    low = max(0.01, lowcut / nyq)
    high = min(0.99, highcut / nyq)
    
    if low >= high:
        return signal
    
    try:
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal)
    except Exception as e:
        print(f"滤波器错误: {e}")
        return signal

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
    
    total_energy = np.sum(fft_vals)
    peak_energy = fft_vals[max_idx]
    quality = min(1.0, peak_energy / total_energy * 3)
    
    return hr, quality

def get_spo2_hr(ts):
    diff = abs(spo2_df['Timestamp'] - ts)
    idx = diff.idxmin()
    return spo2_df.loc[idx, 'Pulse Rate (bpm)']

class HeartRateSmoother:
    def __init__(self, alpha=SMOOTH_ALPHA, median_window=MEDIAN_WINDOW, max_change=MAX_HR_CHANGE):
        self.alpha = alpha
        self.median_window = median_window
        self.max_change = max_change
        self.ema_hr = 70.0
        self.median_history = []
        self.last_valid = 70.0
    
    def update(self, new_hr, quality):
        if quality < 0.3:
            return self.last_valid
        
        if new_hr < MIN_HR or new_hr > MAX_HR:
            return self.last_valid
        
        if abs(new_hr - self.last_valid) > self.max_change:
            new_hr = (new_hr + 2 * self.last_valid) / 3
        
        self.ema_hr = self.alpha * new_hr + (1 - self.alpha) * self.ema_hr
        
        self.median_history.append(self.ema_hr)
        if len(self.median_history) > self.median_window:
            self.median_history.pop(0)
        
        smoothed_hr = np.median(self.median_history)
        self.last_valid = smoothed_hr
        
        return smoothed_hr

# ============ 主处理流程 ============
def main():
    print("="*50)
    print("rPPG心率监测系统 - 参数配置")
    print("="*50)
    print(f"低通滤波器参数: LOWCUT={LOWCUT}Hz, HIGHCUT={HIGHCUT}Hz, ORDER={FILTER_ORDER}")
    print(f"平滑参数: ALPHA={SMOOTH_ALPHA}, MAX_CHANGE={MAX_HR_CHANGE}bpm, MEDIAN_WINDOW={MEDIAN_WINDOW}")
    print(f"信号质量阈值: {SIGNAL_STD_THRESHOLD}")
    print("="*50)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件: {VIDEO_PATH}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
        print(f"警告: 无法获取帧率，使用默认值 {fps} fps")
    
    print(f"视频信息: 帧率={fps:.2f} fps")
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("错误: 无法加载面部检测器")
        cap.release()
        return
    
    signals = {
        'forehead': [],
        'left_cheek': [],
        'right_cheek': []
    }
    results = []
    frame_count = 0
    
    hr_smoother = HeartRateSmoother()
    
    # 初始化图表 - 只保留心率曲线
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    line1, = ax.plot([], [], 'b-', label="rPPG HR")
    line2, = ax.plot([], [], 'r-', label="Spo2 HR")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heart Rate (bpm)")
    ax.set_xlim(0, 60)
    ax.set_ylim(40, 120)
    ax.grid(True)
    ax.legend()
    
    last_time = time.time()
    frame_duration = 1.0 / fps
    
    filter_history = []
    base_ts = spo2_df['Timestamp'].iloc[0]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        elapsed_sec = frame_count / fps
        video_ts = base_ts + timedelta(seconds=elapsed_sec)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(100, 100))

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # 额头ROI
            fh_x = max(0, x + int(0.25 * w))
            fh_y = max(0, y + int(0.05 * h))
            fh_w = min(frame.shape[1] - fh_x, int(0.5 * w))
            fh_h = min(frame.shape[0] - fh_y, int(0.15 * h))
            
            if fh_w > 5 and fh_h > 5:
                forehead_roi = frame[fh_y:fh_y+fh_h, fh_x:fh_x+fh_w]
                cv2.rectangle(frame, (fh_x, fh_y), (fh_x+fh_w, fh_y+fh_h), (255, 0, 0), 2)
                signals['forehead'].append(np.mean(forehead_roi[:, :, 1]))
            else:
                signals['forehead'].append(0)

            # 左脸颊ROI
            lc_x = max(0, x + int(0.2 * w))
            lc_y = max(0, y + int(0.4 * h))
            lc_w = min(frame.shape[1] - lc_x, int(0.15 * w))
            lc_h = min(frame.shape[0] - lc_y, int(0.36 * h))
            
            if lc_w > 5 and lc_h > 5:
                left_cheek_roi = frame[lc_y:lc_y+lc_h, lc_x:lc_x+lc_w]
                cv2.rectangle(frame, (lc_x, lc_y), (lc_x+lc_w, lc_y+lc_h), (0, 255, 255), 2)
                signals['left_cheek'].append(np.mean(left_cheek_roi[:, :, 1]))
            else:
                signals['left_cheek'].append(0)

            # 右脸颊ROI
            rc_x = max(0, x + int(0.65 * w))
            rc_y = max(0, y + int(0.4 * h))
            rc_w = min(frame.shape[1] - rc_x, int(0.15 * w))
            rc_h = min(frame.shape[0] - rc_y, int(0.36 * h))
            
            if rc_w > 5 and rc_h > 5:
                right_cheek_roi = frame[rc_y:rc_y+rc_h, rc_x:rc_x+rc_w]
                cv2.rectangle(frame, (rc_x, rc_y), (rc_x+rc_w, rc_y+rc_h), (0, 255, 255), 2)
                signals['right_cheek'].append(np.mean(right_cheek_roi[:, :, 1]))
            else:
                signals['right_cheek'].append(0)

            if len(signals['forehead']) >= int(fps * 10):
                window_len = int(fps * 10)
                min_len = min(len(signals['forehead']), len(signals['left_cheek']), len(signals['right_cheek']))
                if min_len < window_len:
                    continue
                    
                roi_array = np.array([
                    signals['forehead'][-window_len:],
                    signals['left_cheek'][-window_len:],
                    signals['right_cheek'][-window_len:]
                ]).T

                if METHOD == 'green':
                    target_signal = bandpass_filter(signals['forehead'][-window_len:], fps)
                    signal_std = np.std(signals['forehead'][-window_len:])
                elif METHOD == 'pca':
                    pca = PCA(n_components=1)
                    comp = pca.fit_transform(roi_array)
                    target_signal = bandpass_filter(comp.flatten(), fps)
                    signal_std = np.std(comp.flatten())
                elif METHOD == 'fusion':
                    filtered_fh = bandpass_filter(signals['forehead'][-window_len:], fps)
                    filtered_lc = bandpass_filter(signals['left_cheek'][-window_len:], fps)
                    filtered_rc = bandpass_filter(signals['right_cheek'][-window_len:], fps)
                    target_signal = np.mean([filtered_fh, filtered_lc, filtered_rc], axis=0)
                    signal_std = np.mean([np.std(filtered_fh), np.std(filtered_lc), np.std(filtered_rc)])
                else:
                    target_signal = bandpass_filter(signals['forehead'][-window_len:], fps)
                    signal_std = np.std(signals['forehead'][-window_len:])

                hr, quality = robust_estimate_hr_fft(target_signal, fps)
                quality *= min(1.0, signal_std / SIGNAL_STD_THRESHOLD)
                spo2_hr = get_spo2_hr(video_ts)
                smoothed_hr = hr_smoother.update(hr, quality)
                
                filter_history.append({
                    'time': elapsed_sec,
                    'raw_hr': hr,
                    'smoothed_hr': smoothed_hr,
                    'spo2_hr': spo2_hr,
                    'quality': quality,
                    'signal_std': signal_std
                })
                
                hr_to_display = smoothed_hr
                results.append({
                    'time': elapsed_sec, 
                    'rppg_hr': hr_to_display, 
                    'spo2_hr': spo2_hr
                })

                # 更新画面文本
                cv2.putText(frame, f"rPPG HR: {hr_to_display:.1f} bpm", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, f"Spo2 HR: {spo2_hr:.1f} bpm", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # # 根据质量显示不同颜色
                # color = (0, 255, 0) if quality > 0.5 else (0, 165, 255) if quality > 0.3 else (0, 0, 255)
                # cv2.putText(frame, f"Quality: {quality:.2f}", (10, 90),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                cv2.putText(frame, f"Method: {METHOD}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # # 显示当前滤波器参数
                # cv2.putText(frame, f"Filter: LP={HIGHCUT}Hz", (10, 150),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                # 实时更新折线图 - 只保留心率曲线
                if results:
                    times = [r['time'] for r in results]
                    rppg_hrs = [r['rppg_hr'] for r in results]
                    spo2_hrs = [r['spo2_hr'] for r in results]
                    
                    line1.set_data(times, rppg_hrs)
                    line2.set_data(times, spo2_hrs)
                    
                    # 自动调整X轴范围
                    if times:
                        ax.set_xlim(max(0, times[0]), min(120, times[-1] + 10))
                    
                    ax.relim()
                    ax.autoscale_view()
                    fig.canvas.draw()
                    fig.canvas.flush_events()

        # 显示帧
        display_frame = cv2.resize(frame, (800, 600))
        window_name = f"rPPG HR Detection - LP={HIGHCUT}Hz"
        cv2.imshow(window_name, display_frame)
        cv2.moveWindow(window_name, 100, 190)  # x=100，y=130，往下移动一点


        # 控制帧率播放
        elapsed = time.time() - last_time
        wait_time = max(0, frame_duration - elapsed)
        key = cv2.waitKey(max(1, int(wait_time * 1000))) & 0xFF
        if key == ord('q'):
            break
            
        last_time = time.time()

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    
    # 保存详细分析数据
    if filter_history:
        df = pd.DataFrame(filter_history)
        df.to_csv('hr_analysis.csv', index=False)
        print("详细心率分析已保存到 hr_analysis.csv")
        
        # 绘制心率对比图 - 只保留心率曲线
        plt.figure(figsize=(12, 6))
        plt.plot(df['time'], df['raw_hr'], 'r-', alpha=0.5, label='原始心率')
        plt.plot(df['time'], df['smoothed_hr'], 'b-', label='平滑心率')
        plt.plot(df['time'], df['spo2_hr'], 'g-', label='血氧计心率')
        plt.xlabel('时间 (秒)')
        plt.ylabel('心率 (BPM)')
        plt.title(f'心率对比 - 低通滤波: {HIGHCUT}Hz')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'hr_comparison_LP_{HIGHCUT}Hz.png')
        plt.show()

if __name__ == "__main__":
    main()