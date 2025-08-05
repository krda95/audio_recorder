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
import matplotlib.ticker as ticker
import mplcursors
import numpy as np

parser = argparse.ArgumentParser(description="Rejestrator dźwięku z wykresem")
parser.add_argument('--czas', type=str, help='Czas nagrania z jednostką: np. 10s, 5m, 2h')
parser.add_argument('--plik', type=str, help='Wczytaj dane z pliku CSV i pokaż wykres')
parser.add_argument('--prog', type=float, default=None, help='Próg dBFS do zapisu (jeśli podany, zapisuje tylko głośne fragmenty)')
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
        # Final plotting section, adapted for loaded file
        start_datetime = datetime.strptime(df['time'].iloc[0], "%H:%M:%S.%f")
        seconds_from_start = [
            (datetime.strptime(t, "%H:%M:%S.%f") - start_datetime).total_seconds()
            for t in df['time']
        ]
        x_seconds = seconds_from_start

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        ax1.plot(x_seconds, df['rms'], label='RMS', color='blue')
        ax1.set_ylabel("RMS")
        ax1.set_ylim(0, df['rms'].max() * 1.1 + 1e-6)
        ax1.grid(True)
        ax1.legend()

        ax2.plot(x_seconds, df['dbfs'], label='dBFS', color='red')
        ax2.set_ylabel("dBFS")
        ax2.set_ylim(-90, 0)
        ax2.set_xlabel("Czas")

        # Primary axis: one label per full second
        max_sec = int(np.floor(seconds_from_start[-1]))
        primary_locs = []
        primary_labels = []
        for s in range(0, max_sec + 1):
            # find index for this second
            idx = next((i for i, v in enumerate(seconds_from_start) if v >= s), len(seconds_from_start) - 1)
            primary_locs.append(x_seconds[idx])
            primary_labels.append(df['time'].iloc[idx])
        ax2.set_xticks(primary_locs)
        ax2.set_xticklabels(primary_labels, rotation=90)

        ax2.grid(True)
        ax2.legend()
        if args.prog is not None:
            ax2.axhline(args.prog, color='green', linestyle='--', label=f'Próg {args.prog} dBFS')
            ax2.legend()

        def format_seconds(x, pos):
            total_seconds = int(round(x))
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            hours = minutes // 60
            if hours > 0:
                return f"{hours}h {minutes}m {seconds}s"
            if minutes > 0:
                return f"{minutes}m {seconds}s"
            else:
                return f"{seconds}s"

        ax2_sec = ax2.secondary_xaxis('top')
        ax2_sec.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: format_seconds(x, pos)))
        ax2_sec.set_xlabel("Czas od startu")
        for label in ax2_sec.get_xticklabels():
            label.set_rotation(90)
        ax2_sec.set_xticks(list(range(0, max_sec + 1)))

        plt.suptitle("Pełny wykres nagrania")
        plt.tight_layout()
        # Enable interactive hover tooltips showing relative seconds, timestamp, and value
        cursor = mplcursors.cursor(hover=True)
        @cursor.connect("add")
        def on_add(sel):
            x, y = sel.target
            # Determine nearest index for timestamp lookup
            idx = int(round(x))
            idx = max(0, min(idx, len(df) - 1))
            timestamp = df['time'].iloc[idx]
            sel.annotation.set_text(
                f"Rel: {x:.2f}s\nTime: {timestamp}\nValue: {y:.2f}"
            )
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
    amplitude = np.linalg.norm(indata)
    rms = np.sqrt(np.mean(indata**2))
    dbfs = 20 * np.log10(rms + 1e-12)
    timestamp_seconds = time.time()
    timestamp_str = datetime.fromtimestamp(timestamp_seconds).strftime("%H:%M:%S.%f")[:-3]  # format HH:MM:SS.mmm
    rel_time = timestamp_seconds - start_time
    audio_buffer.append(indata.copy()) # mozna filtrowac dodajac do warunku ponizej
    if args.prog is None or dbfs >= args.prog:
        data_list.append((timestamp_str, amplitude, dbfs))
    x_vals.append(rel_time)
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
# for label in ax_dbfs.get_xticklabels():
#     label.set_rotation(90)
ax_dbfs.grid(True)
ax_dbfs.legend()
if args.prog is not None:
    ax_dbfs.axhline(args.prog, color='green', linestyle='--', label=f'Próg {args.prog} dBFS')
    ax_dbfs.legend()

print(f"Nagrywanie przez {DURATION_SEC} sekund...")
if args.prog is not None:
    print(f"Tylko fragmenty powyżej progu {args.prog} dBFS będą zapisywane.")

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

if not data_list:
    print("Brak danych powyżej progu — nic nie zapisano.")
    sys.exit(0)

df = pd.DataFrame(data_list, columns=['time', 'rms', 'dbfs'])
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

start_datetime = datetime.strptime(df['time'].iloc[0], "%H:%M:%S.%f")
seconds_from_start = [
    (datetime.strptime(t, "%H:%M:%S.%f") - start_datetime).total_seconds()
    for t in df['time']
]
x_seconds = seconds_from_start

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

ax1.plot(x_seconds, df['rms'], label='RMS', color='blue')
ax1.set_ylabel("RMS")
ax1.set_ylim(0, df['rms'].max() * 1.1 + 1e-6)
ax1.grid(True)
ax1.legend()

ax2.plot(x_seconds, df['dbfs'], label='dBFS', color='red')
ax2.set_ylabel("dBFS")
ax2.set_ylim(-90, 0)
ax2.set_xlabel("Czas")


# Primary axis: one label per full second
max_sec = int(np.floor(seconds_from_start[-1]))
primary_locs = []
primary_labels = []
for s in range(0, max_sec + 1):
    # find index for this second
    idx = next((i for i, v in enumerate(seconds_from_start) if v >= s), len(seconds_from_start) - 1)
    primary_locs.append(x_seconds[idx])
    primary_labels.append(df['time'].iloc[idx])
ax2.set_xticks(primary_locs)
ax2.set_xticklabels(primary_labels, rotation=90)

ax2.grid(True)
ax2.legend()
if args.prog is not None:
    ax2.axhline(args.prog, color='green', linestyle='--', label=f'Próg {args.prog} dBFS')
    ax2.legend()

def format_seconds(x, pos):
    total_seconds = int(round(x))
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    hours = minutes // 60
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

ax2_sec = ax2.secondary_xaxis('top')
ax2_sec.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: format_seconds(x, pos)))
ax2_sec.set_xlabel("Czas od startu")
for label in ax2_sec.get_xticklabels():
    label.set_rotation(90)
ax2_sec.set_xticks(list(range(0, max_sec + 1)))

plt.suptitle("Pełny wykres nagrania")
plt.tight_layout()
# Enable interactive hover tooltips showing relative seconds, timestamp, and value
cursor = mplcursors.cursor(hover=True)
@cursor.connect("add")
def on_add(sel):
    x, y = sel.target
    # Determine nearest index for timestamp lookup
    idx = int(round(x))
    idx = max(0, min(idx, len(df) - 1))
    timestamp = df['time'].iloc[idx]
    sel.annotation.set_text(
        f"Rel: {x:.2f}s\nTime: {timestamp}\nValue: {y:.2f}"
    )
plt.show()