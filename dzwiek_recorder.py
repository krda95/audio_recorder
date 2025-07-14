import sounddevice as sd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
import argparse
from pydub import AudioSegment
import sys
import re

parser = argparse.ArgumentParser(description="Rejestrator dźwięku z wykresem")
parser.add_argument('--czas', type=str, help='Czas nagrania z jednostką: np. 10s, 5m, 2h')
parser.add_argument('--plik', type=str, help='Wczytaj dane z pliku CSV i pokaż wykres')
args = parser.parse_args()


def parse_duration(value):
    match = re.match(r'^(\d+)([smh])$', value)
    if not match:
        print("Błąd: czas musi mieć format np. 10s, 5m, 2h")
        sys.exit(1)
    liczba, jednostka = int(match.group(1)), match.group(2)
    if jednostka == 's':
        return liczba
    elif jednostka == 'm':
        return liczba * 60
    elif jednostka == 'h':
        return liczba * 3600


if args.plik:
    try:
        df = pd.read_csv(args.plik)
        if 'rms' in df.columns and 'dbfs' in df.columns:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            ax1.plot(df['timestamp'], df['rms'], label='RMS', color='blue')
            ax1.set_ylabel("RMS")
            ax1.set_ylim(0, df['rms'].max() * 1.1 + 1e-6)
            ax1.grid(True)
            ax1.legend()
            ax2.plot(df['timestamp'], df['dbfs'], label='dBFS', color='red')
            ax2.set_ylabel("dBFS")
            ax2.set_ylim(-90, 0)
            ax2.set_xlabel("Czas (s)")
            ax2.grid(True)
            ax2.legend()
            plt.suptitle(f"Wykres z pliku: {args.plik}")
            plt.tight_layout()
            plt.show()
        else:
            plt.plot(df['timestamp'], df['amplitude'])
            plt.xlabel("Czas (s)")
            plt.ylabel("Amplituda")
            plt.title(f"Wykres z pliku: {args.plik}")
            plt.grid()
            plt.show()
        sys.exit(0)
    except Exception as e:
        print(f"Błąd podczas wczytywania pliku: {e}")
        sys.exit(1)

DURATION_SEC = parse_duration(args.czas)
SAMPLERATE = 44100
CHUNK_SIZE = 1024
CHANNELS = 1

timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
CSV_FILENAME = f"baza/amplituda_{timestamp_str}.csv"
MP3_FILENAME = f"baza/dzwiek_{timestamp_str}.mp3"

audio_buffer = []
data_list = []

x_vals, rms_vals, dbfs_vals = [], [], []

def audio_callback(indata, frames, time_info, status):
    audio_buffer.append(indata.copy())
    amplitude = np.linalg.norm(indata)
    rms = np.sqrt(np.mean(indata**2))
    dbfs = 20 * np.log10(rms + 1e-12)
    timestamp = time.time() - start_time
    data_list.append((timestamp, amplitude, dbfs))
    x_vals.append(timestamp)
    rms_vals.append(amplitude)
    dbfs_vals.append(dbfs)

plt.ion()
fig, (ax_rms, ax_dbfs) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

line_rms, = ax_rms.plot([], [], label='RMS', color='blue')
ax_rms.set_ylabel("RMS")
limit = max(rms_vals[-500:]) * 1.1 + 1e-6 if rms_vals else 0.1
ax_rms.set_ylim(0, limit)
ax_rms.grid(True)
ax_rms.legend()

line_dbfs, = ax_dbfs.plot([], [], label='dBFS', color='red')
ax_dbfs.set_ylabel("dBFS")
ax_dbfs.set_ylim(-90, 0)
ax_dbfs.set_xlabel("Czas (s)")
ax_dbfs.grid(True)
ax_dbfs.legend()

print(f"Nagrywanie przez {DURATION_SEC} sekund...")

with sd.InputStream(callback=audio_callback, channels=CHANNELS, samplerate=SAMPLERATE, blocksize=CHUNK_SIZE):
    start_time = time.time()
    while time.time() - start_time < DURATION_SEC:
        if x_vals:
            timestamp = x_vals[-1]
            line_rms.set_data(x_vals[-500:], rms_vals[-500:])
            line_dbfs.set_data(x_vals[-500:], dbfs_vals[-500:])
            ax_rms.set_xlim(max(0, timestamp - 10), timestamp + 1)
            ax_dbfs.set_xlim(max(0, timestamp - 10), timestamp + 1)
            ax_rms.set_ylim(0, max(rms_vals[-500:]) * 1.1 + 1e-6)
            ax_dbfs.set_ylim(min(dbfs_vals[-500:]) - 5, 0)
            fig.canvas.draw()
            fig.canvas.flush_events()
        time.sleep(0.01)

plt.ioff()
plt.close()

df = pd.DataFrame(data_list, columns=['timestamp', 'rms', 'dbfs'])
df.to_csv(CSV_FILENAME, index=False)
print(f"Zapisano dane do {CSV_FILENAME}")

audio_data = np.concatenate(audio_buffer, axis=0)
audio_int16 = np.int16(audio_data * 32767)

audio_segment = AudioSegment(
    data=audio_int16.tobytes(),
    sample_width=2,
    frame_rate=SAMPLERATE,
    channels=1
)

audio_segment.export(MP3_FILENAME, format="mp3", bitrate="192k")
print(f"Zapisano dźwięk do pliku MP3: {MP3_FILENAME}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

ax1.plot(df['timestamp'], df['rms'], label='RMS', color='blue')
ax1.set_ylabel("RMS")
ax1.set_ylim(0, df['rms'].max() * 1.1 + 1e-6)
ax1.grid(True)
ax1.legend()

ax2.plot(df['timestamp'], df['dbfs'], label='dBFS', color='red')
ax2.set_ylabel("dBFS")
ax2.set_ylim(-90, 0)
ax2.set_xlabel("Czas (s)")
ax2.grid(True)
ax2.legend()

plt.suptitle("Pełny wykres nagrania")
plt.tight_layout()
plt.show()