import os
import tkinter as tk
from tkinter import filedialog
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
from matplotlib.widgets import SpanSelector, Button
import mplcursors
import numpy as np

parser = argparse.ArgumentParser(description="Rejestrator dźwięku z wykresem")
parser.add_argument('--czas', type=str, default='10s', help='Czas nagrania z jednostką: np. 10s, 5m, 2h (domyślnie 10s)')
parser.add_argument('--prog', type=float, default=None, help='Próg dBFS do zapisu (jeśli podany, zapisuje tylko głośne fragmenty)')
parser.add_argument('--plik', action='store_true', help='Otwórz okno wyboru pliku CSV')
args = parser.parse_args()


interrupted = False
ignored = False

def parse_duration(value):
    match = re.match(r'^(\d+)([smh]?)$', str(value))
    if not match:
        print("Błąd: czas musi mieć format np. 10s, 5m, 2h lub liczba sekund (np. 10)")
        sys.exit(1)
    liczba, jednostka = int(match.group(1)), (match.group(2) or 's')
    if jednostka == 's':
        return liczba
    elif jednostka == 'm':
        return liczba * 60
    elif jednostka == 'h':
        return liczba * 3600

def pick_csv_via_finder() -> str:
    root = tk.Tk()
    root.withdraw()
    root.update()
    initdir = os.path.join(os.getcwd(), 'baza') if os.path.isdir(os.path.join(os.getcwd(), 'baza')) else os.getcwd()
    filepath = filedialog.askopenfilename(
        title='Wybierz plik CSV',
        initialdir=initdir,
        filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
    )
    root.destroy()
    return filepath if filepath else None

def format_seconds(x):
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

if args.plik:
    picked = pick_csv_via_finder()
    if picked:
        print(f"Wybrano plik: {picked}")
        args.plik = picked
    else:
        print("Nie wybrano pliku – wyłączam program.")
        sys.exit(-1)

if args.plik:
    try:
        df = pd.read_csv(args.plik)
        if 'time' in df.columns and 'rms' in df.columns and 'dbfs' in df.columns:
            start_datetime = datetime.strptime(df['time'].iloc[0], "%H:%M:%S.%f")
            seconds_from_start = [
                (datetime.strptime(t, "%H:%M:%S.%f") - start_datetime).total_seconds()
                for t in df['time']
            ]
            x_seconds = seconds_from_start

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            # RMS
            ax1.plot(x_seconds, df['rms'], label='RMS', color='blue')
            ax1.set_ylabel("RMS")
            ax1.set_ylim(0, df['rms'].max() * 1.1 + 1e-6)
            ax1.grid(True)
            ax1.legend()
            # dBFS
            ax2.plot(x_seconds, df['dbfs'], label='dBFS', color='red')
            ax2.set_ylabel("dBFS")
            ax2.set_ylim(-90, 0)
            ax2.set_xlabel("Czas")

            max_sec = int(np.floor(seconds_from_start[-1]))
            primary_locs, primary_labels = [], []
            for s in range(max_sec + 1):
                idx = next((i for i, v in enumerate(seconds_from_start) if v >= s), len(seconds_from_start)-1)
                primary_locs.append(x_seconds[idx])
                primary_labels.append(df['time'].iloc[idx])
            ax2.set_xticks(primary_locs)
            ax2.set_xticklabels(primary_labels, rotation=90)

            ax2.grid(True)
            ax2.legend()
            if args.prog is not None:
                ax2.axhline(args.prog, color='green', linestyle='--', label=f'Próg {args.prog} dBFS')
                ax2.legend()

            ax2_sec = ax2.secondary_xaxis('top')
            ax2_sec.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: format_seconds(x)))
            ax2_sec.set_xlabel("Czas od startu")
            ax2_sec.set_xticks(list(range(max_sec+1)))
            for label in ax2_sec.get_xticklabels():
                label.set_rotation(90)

            cursor = mplcursors.cursor(hover=True)
            @cursor.connect("add")
            def on_add(sel):
                x, y = sel.target
                idx = max(0, min(len(df)-1, int(round(x))))
                sel.annotation.set_text(
                    f"Rel: {x:.2f}s\nTime: {df['time'].iloc[idx]}\nValue: {y:.2f}"
                )

            plt.suptitle("Pełny wykres nagrania")
            plt.tight_layout()
            def onselect(xmin, xmax):
                ax1.set_xlim(xmin, xmax)
                ax2.set_xlim(xmin, xmax)
                fig.canvas.draw()

            span = SpanSelector(ax2, onselect, 'horizontal', useblit=True,
                                props=dict(alpha=0.3, facecolor='grey'))

            orig_xlim1 = ax1.get_xlim()
            orig_xlim2 = ax2.get_xlim()

            def reset_zoom(event):
                if event.key == 'r':
                    ax1.set_xlim(orig_xlim1)
                    ax2.set_xlim(orig_xlim2)
                    fig.canvas.draw()

            fig.canvas.mpl_connect('key_press_event', reset_zoom)

            plt.show()
            sys.exit(0)

        # Legacy CSV with timestamp seconds, RMS and dBFS
        elif 'timestamp' in df.columns and 'rms' in df.columns and 'dbfs' in df.columns:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            
            ax1.plot(df['timestamp'], df['rms'], label='RMS', color='blue')
            ax1.set_ylabel("RMS")
            ax1.set_xlabel("Czas (s)")
            ax1.grid(True)
            ax1.legend()
            
            ax2.plot(df['timestamp'], df['dbfs'], label='dBFS', color='red')
            ax2.set_ylabel("dBFS")
            ax2.set_xlabel("Czas (s)")
            ax2.set_ylim(-90, 0)
            ax2.grid(True)
            if args.prog is not None:
                ax2.axhline(args.prog, color='green', linestyle='--', label=f'Próg {args.prog} dBFS')
            ax2.legend()

            cursor = mplcursors.cursor(hover=True)
            @cursor.connect("add")
            def on_add(sel):
                x, y = sel.target
                idx = max(0, min(len(df)-1, int(round(x))))
                sel.annotation.set_text(
                    f"Time: {df['timestamp'].iloc[idx]}\nValue: {y:.2f}"
                )
            
            plt.tight_layout()
            def onselect(xmin, xmax):
                ax1.set_xlim(xmin, xmax)
                ax2.set_xlim(xmin, xmax)
                fig.canvas.draw()

            span = SpanSelector(ax2, onselect, 'horizontal', useblit=True,
                                props=dict(alpha=0.3, facecolor='grey'))

            orig_xlim1 = ax1.get_xlim()
            orig_xlim2 = ax2.get_xlim()

            def reset_zoom(event):
                if event.key == 'r':
                    ax1.set_xlim(orig_xlim1)
                    ax2.set_xlim(orig_xlim2)
                    fig.canvas.draw()

            fig.canvas.mpl_connect('key_press_event', reset_zoom)

            plt.show()
            sys.exit(0)

        else:
            print("Nie rozpoznano formatu pliku CSV. Sprawdź nagłówki kolumn.")
            sys.exit(1)

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
    timestamp_str = datetime.fromtimestamp(timestamp_seconds).strftime("%H:%M:%S.%f")[:-3]
    rel_time = timestamp_seconds - start_time
    audio_buffer.append(indata.copy())
    if args.prog is None or dbfs >= args.prog:
        data_list.append((timestamp_str, rms, dbfs))
    x_vals.append(rel_time)
    rms_vals.append(rms)
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

stop_ax = fig.add_axes([0.8, 0.01, 0.1, 0.05])
stop_btn = Button(stop_ax, 'STOP & SAVE', hovercolor='tomato')
ignore_ax = fig.add_axes([0.7, 0.01, 0.07, 0.05])
ignore_btn = Button(ignore_ax, 'IGNORE', hovercolor='green')

def _on_stop_clicked(event):
    global interrupted
    interrupted = True
    print('Zatrzymano nagrywanie (przycisk STOP).')

def _on_ignore_clicked(event):
    global ignored
    ignored = True
    print('Zatrzymano nagrywanie (przycisk IGNORE).')

stop_btn.on_clicked(_on_stop_clicked)
ignore_btn.on_clicked(_on_ignore_clicked)


print(f"Nagrywanie przez {DURATION_SEC} sekund...")
if args.prog is not None:
    print(f"Tylko fragmenty powyżej progu {args.prog} dBFS będą zapisywane.")

with sd.InputStream(callback=audio_callback, channels=CHANNELS, samplerate=SAMPLERATE, blocksize=CHUNK_SIZE):
    start_time = time.time()
    while (not ignored and not interrupted) and (time.time() - start_time < DURATION_SEC):
        if x_vals:
            timestamp = x_vals[-1]
            line_rms.set_data(x_vals[-500:], rms_vals[-500:])
            line_dbfs.set_data(x_vals[-500:], dbfs_vals[-500:])
            ax_rms.set_xlim(max(0, timestamp - 10), timestamp + 1)
            ax_dbfs.set_xlim(max(0, timestamp - 10), timestamp + 1)
            ax_rms.set_ylim(0, max(rms_vals[-500:]) * 1.1 + 1e-6)
            bottom = (min(dbfs_vals[-500:]) - 5) if dbfs_vals else -90
            ax_dbfs.set_ylim(bottom, 0)
            fig.canvas.draw()
            fig.canvas.flush_events()
        time.sleep(0.01)

plt.ioff()
plt.close()

if ignored:
    print("Ignored — nic nie zapisano.")
    sys.exit(0)

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


max_sec = int(np.floor(seconds_from_start[-1]))
primary_locs = []
primary_labels = []
for s in range(0, max_sec + 1):
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

ax2_sec = ax2.secondary_xaxis('top')
ax2_sec.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: format_seconds(x)))
ax2_sec.set_xlabel("Czas od startu")
for label in ax2_sec.get_xticklabels():
    label.set_rotation(90)
ax2_sec.set_xticks(list(range(0, max_sec + 1)))

plt.suptitle("Pełny wykres nagrania")
plt.tight_layout()
cursor = mplcursors.cursor(hover=True)
@cursor.connect("add")
def on_add(sel):
    x, y = sel.target
    idx = int(round(x))
    idx = max(0, min(idx, len(df) - 1))
    timestamp = df['time'].iloc[idx]
    sel.annotation.set_text(
        f"Rel: {x:.2f}s\nTime: {timestamp}\nValue: {y:.2f}"
    )

def onselect(xmin, xmax):
    ax1.set_xlim(xmin, xmax)
    ax2.set_xlim(xmin, xmax)
    fig.canvas.draw()

span = SpanSelector(ax2, onselect, 'horizontal', useblit=True,
                    props=dict(alpha=0.3, facecolor='grey'))

orig_xlim1 = ax1.get_xlim()
orig_xlim2 = ax2.get_xlim()

def reset_zoom(event):
    if event.key == 'r':
        ax1.set_xlim(orig_xlim1)
        ax2.set_xlim(orig_xlim2)
        fig.canvas.draw()

fig.canvas.mpl_connect('key_press_event', reset_zoom)

plt.show()