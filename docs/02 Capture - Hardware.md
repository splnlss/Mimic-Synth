---
tags: [build, 02-capture-hardware, capture-rig, hardware, asio, production]
created: 2026-04-19
---

# 02 Capture — Hardware Capture Rig (ASIO + Synth-Agnostic)

> [!info] Goals
> **Synth-agnostic:** the rig takes *any* Bucket 1 profile (Juno-106 SysEx, Peak CC/NRPN, anything future) and captures a dataset without code changes. **Stable:** runs unattended overnight, recovers from dropouts, reports health metrics, verifies every capture. **Low-latency capture via ASIO** on Windows (CoreAudio on macOS, JACK on Linux).

## What changes vs V1

| Concern                | V1 (OB-Xf)                   | V2 (Hardware)                          |
| ---------------------- | ---------------------------- | -------------------------------------- |
| Signal path            | Direct VST automation        | MIDI out → synth → audio in            |
| Synth coupling         | Hard-coded to OB-Xf          | Driven entirely by the profile YAML    |
| Parameter encoding     | VST parameter index          | CC / 14-bit CC / NRPN / SysEx          |
| Reset                  | Instant (write VST state)    | Scripted MIDI sequence with settle time |
| Speed                  | Faster than real-time        | Real-time only (~3 s per capture)      |
| Latency                | N/A                          | Measured and compensated per session    |
| Error modes            | Few                          | Many — must detect + recover           |
| Runtime for 100k samples | ~2 hours                   | ~80 hours (plan for overnight × 3)     |

## Hardware requirements

- **Audio interface with ASIO driver** (Windows) or Core Audio (macOS) or JACK (Linux). Low round-trip latency is nice-to-have, not mandatory — we measure and compensate.
  - Good options: Focusrite Scarlett 2i2 / 4i4, MOTU M2/M4, RME Babyface, Audient iD series. Anything with a native ASIO driver (not ASIO4ALL as primary — though ASIO4ALL works in a pinch).
- **MIDI interface.** Most audio interfaces include DIN MIDI. For USB-MIDI synths (Peak), the synth itself is the interface. For DIN-only synths (Juno-106), you need a dedicated USB-to-DIN box (iConnectivity mio, MOTU micro express, Roland UM-ONE).
- **Cabling.** Balanced TRS from synth main out → interface line in. Keep it short. Ground-loop isolators if you get hum.
- **Monitoring.** Headphones or monitors on the interface's output for ear-sanity-check during setup. Not wired into the capture path.

## Software dependencies

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
# or: source .venv/bin/activate   # macOS/Linux

pip install mido python-rtmidi sounddevice numpy scipy \
            soundfile pyarrow pyyaml tqdm click rich
```

Windows-specific for ASIO:

```bash
# sounddevice uses PortAudio, which speaks ASIO natively on Windows.
# No extra install needed — PortAudio is bundled with sounddevice.
# But you DO need the native ASIO driver from your interface vendor installed.
```

Verify ASIO is visible:

```python
import sounddevice as sd
print(sd.query_hostapis())
# You should see an entry with name='ASIO' (Windows only)
sd.query_devices()
# Interface should appear with hostapi matching the ASIO index
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              s02_capture/capture_v2.py                          │
│                                                                 │
│   ┌───────────┐   ┌──────────────┐   ┌─────────────────────┐    │
│   │  Profile  │──▶│   Capture    │──▶│  Storage (Parquet + │    │
│   │  (YAML)   │   │    Loop      │   │   WAV, hash-keyed)  │    │
│   └───────────┘   └──────┬───────┘   └─────────────────────┘    │
│                          │                                       │
│                ┌─────────┴─────────┐                             │
│                │                   │                             │
│          ┌─────▼────┐       ┌──────▼──────┐                      │
│          │   MIDI   │       │  ASIO Audio │                      │
│          │  Sender  │       │  Capturer   │                      │
│          └─────┬────┘       └──────┬──────┘                      │
│                │                   │                             │
└────────────────┼───────────────────┼─────────────────────────────┘
                 │                   │
                 ▼                   ▲
         ┌────────────┐       ┌──────────────┐
         │  Hardware  │──────▶│    Audio     │
         │   Synth    │       │  Interface   │
         └────────────┘       └──────────────┘
```

The key property: the **MIDI Sender** module is the only place that knows how to dispatch CC / NRPN / SysEx. Everything else sees opaque "apply param" calls driven by the profile.

## Project layout

```
MimicSynth/
├── s01_profiles/
│   ├── juno_106.yaml
│   ├── novation_peak.yaml
│   └── obxf.yaml
├── s02_capture/
│   ├── capture_v1.py         # VST capture (software synths)
│   ├── capture_v2.py         # hardware capture (this doc)
│   ├── data/                 # output: samples.parquet + wav/
│   ├── midi_sender.py        # CC / NRPN / SysEx dispatcher
│   ├── audio_capturer.py     # ASIO / CoreAudio / JACK backends
│   ├── latency.py            # round-trip calibration
│   ├── capture_loop.py       # orchestration
│   └── health.py             # monitoring + metrics
├── s03_dataset/
└── config/
    └── session.yaml          # audio device name, MIDI port, output dir
```

## Synth profile schema (extended)

V2 uses the same profile format as V1 but with a richer parameter encoding vocabulary:

```yaml
synth:
  id: <string>
  name: <string>
  transport: <din_midi | usb_midi | vst_host>
  channel: <1-16>              # ignored for vst_host
  parameter_encoding: <cc | sysex | nrpn | vst_automation | mixed>

parameters:
  <param_name>:
    encoding: <cc | cc_14bit | nrpn | sysex | vst>
    # encoding-specific fields:
    cc: <0-127>                          # when encoding == cc
    cc_msb: <0-127>                      # when encoding == cc_14bit
    cc_lsb: <0-127>
    nrpn_msb: <0-127>                    # when encoding == nrpn
    nrpn_lsb: <0-127>
    sysex_template: "F0 41 32 00 {pp} {vv} F7"   # when encoding == sysex
    param_num: <int>                     # substituted into {pp}
    range: [min, max]
    continuous: <bool>
    log_scale: <bool>
    categories: [...]                    # when continuous == false
    importance: <0-1>

probe:
  notes: [int, ...]
  velocity: <0-127>
  hold_ms: <int>
  release_ms: <int>

reset:
  messages:
    - {type: cc, cc: <n>, value: <v>}
    - {type: cc_14bit, cc_msb: <n>, cc_lsb: <n>, value: <0-16383>}
    - {type: nrpn, nrpn_msb: <n>, nrpn_lsb: <n>, value: <0-16383>}
    - {type: sysex, hex: "F0 ..."}
    - {type: sysex_file, path: <path>}
    - {type: sleep, ms: <n>}
  settle_ms: <int>

audio:
  expected_input_db: -18           # target RMS for healthy captures
  silence_floor_db: -60            # below this = consider silent
```

## MIDI Sender module

`synthmimic/midi_sender.py` — the only place in the codebase with knowledge of CC / NRPN / SysEx encoding.

```python
"""Synth-agnostic MIDI dispatcher. Driven by the profile's encoding field."""
import mido
import time

class MidiSender:
    def __init__(self, port_name, channel=1):
        self.port = mido.open_output(port_name)
        self.channel = channel - 1   # mido uses 0-indexed

    def send_cc(self, cc, value):
        self.port.send(mido.Message("control_change",
                                    channel=self.channel,
                                    control=cc, value=int(value)))

    def send_cc_14bit(self, msb_cc, lsb_cc, value):
        """14-bit CC: value in [0, 16383], sent as two CCs."""
        msb = (int(value) >> 7) & 0x7F
        lsb = int(value) & 0x7F
        self.send_cc(msb_cc, msb)
        self.send_cc(lsb_cc, lsb)

    def send_nrpn(self, nrpn_msb, nrpn_lsb, value):
        """Standard NRPN with 14-bit data value."""
        msb = (int(value) >> 7) & 0x7F
        lsb = int(value) & 0x7F
        self.send_cc(99, nrpn_msb)
        self.send_cc(98, nrpn_lsb)
        self.send_cc(6, msb)
        self.send_cc(38, lsb)
        # NRPN null to prevent stray data edits
        self.send_cc(99, 127)
        self.send_cc(98, 127)

    def send_sysex(self, hex_string):
        """Accepts 'F0 ... F7' with spaces."""
        data = bytes.fromhex(hex_string.replace(" ", ""))
        assert data[0] == 0xF0 and data[-1] == 0xF7
        # mido wants the body without F0/F7
        self.port.send(mido.Message("sysex", data=data[1:-1]))

    def send_sysex_file(self, path):
        for msg in mido.read_syx_file(path):
            self.port.send(msg)

    def note_on(self, note, velocity=100):
        self.port.send(mido.Message("note_on", channel=self.channel,
                                    note=note, velocity=velocity))

    def note_off(self, note):
        self.port.send(mido.Message("note_off", channel=self.channel,
                                    note=note, velocity=0))

    def all_notes_off(self):
        self.send_cc(123, 0)
```

## Applying a parameter vector

Agnostic dispatcher — reads the profile's `encoding` field and routes to the right sender method:

```python
def apply_params(sender, params_dict, profile):
    """params_dict: {param_name: normalised_value_in_[0,1]}"""
    for name, norm_value in params_dict.items():
        spec = profile["parameters"][name]
        lo, hi = spec["range"]

        # Map normalised [0,1] → profile range, respecting log_scale
        if spec.get("log_scale"):
            import math
            v = lo * (hi / lo) ** norm_value if lo > 0 else lo + (hi - lo) * norm_value
        else:
            v = lo + (hi - lo) * norm_value

        enc = spec["encoding"]
        if enc == "cc":
            sender.send_cc(spec["cc"], int(round(v)))
        elif enc == "cc_14bit":
            sender.send_cc_14bit(spec["cc_msb"], spec["cc_lsb"], int(round(v)))
        elif enc == "nrpn":
            sender.send_nrpn(spec["nrpn_msb"], spec["nrpn_lsb"], int(round(v)))
        elif enc == "sysex":
            # Substitute {pp} and {vv} into the hex template
            pp = f"{spec['param_num']:02X}"
            vv = f"{int(round(v)):02X}"
            hex_str = spec["sysex_template"].format(pp=pp, vv=vv)
            sender.send_sysex(hex_str)
        else:
            raise ValueError(f"Unknown encoding: {enc}")
```

## Audio capturer with ASIO

`synthmimic/audio_capturer.py`:

```python
"""Low-latency audio capture. Uses ASIO on Windows, CoreAudio on macOS, JACK on Linux."""
import sounddevice as sd
import numpy as np

class AudioCapturer:
    def __init__(self, device_name, sample_rate=48000, channels=1,
                 block_size=256, dtype="float32"):
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        self.dtype = dtype
        self.device = self._resolve_device(device_name)

    def _resolve_device(self, name):
        for i, d in enumerate(sd.query_devices()):
            if name.lower() in d["name"].lower() and d["max_input_channels"] > 0:
                return i
        raise RuntimeError(f"No input device matching '{name}'")

    def capture(self, duration_sec):
        n = int(duration_sec * self.sample_rate)
        audio = sd.rec(n, samplerate=self.sample_rate, channels=self.channels,
                       dtype=self.dtype, device=self.device, blocking=True)
        return audio[:, 0] if self.channels == 1 else audio
```

Pick a small `block_size` for lower latency (64–256 samples). Check your interface's ASIO control panel and match.

## Latency calibration utility

`s02_capture/calibrate_latency.py` — sends a short MIDI click, captures the return, finds the offset by cross-correlation. Run once per session before the main capture.

```python
"""Measure MIDI-out → audio-in round-trip latency."""
import sys, time, yaml
import numpy as np
import soundfile as sf
from scipy.signal import correlate
from synthmimic.midi_sender import MidiSender
from synthmimic.audio_capturer import AudioCapturer

SESSION = yaml.safe_load(open("config/session.yaml"))
sender = MidiSender(SESSION["midi_port"], channel=SESSION["midi_channel"])
cap = AudioCapturer(SESSION["audio_device"], sample_rate=48000)

# Play a high, bright note to maximise transient
PROBE_NOTE = 84
# Start capture, then send note-on
import threading
audio_buf = []
def capture():
    audio_buf.append(cap.capture(1.0))

t = threading.Thread(target=capture)
t.start()
time.sleep(0.05)   # let the stream open
t0 = time.time()
sender.note_on(PROBE_NOTE, 127)
t1 = time.time()
t.join()
sender.note_off(PROBE_NOTE)

audio = audio_buf[0]
# Find the first sample where |audio| crosses threshold
abs_audio = np.abs(audio)
threshold = abs_audio.max() * 0.2
onset_sample = int(np.argmax(abs_audio > threshold))
onset_time_sec = onset_sample / 48000

# Midi was sent at t0-t1; we started capturing ~50ms earlier at t0-0.05
# So audio onset's position in the buffer minus ~50ms = latency
latency_ms = onset_time_sec * 1000 - 50
print(f"Measured round-trip latency: {latency_ms:.1f} ms")
# Write to session config so the capture loop can compensate
SESSION["measured_latency_ms"] = latency_ms
yaml.safe_dump(SESSION, open("config/session.yaml", "w"))
```

Typical values: 5–20 ms on a well-tuned ASIO setup. Over 50 ms → something's wrong (buffer size too high, USB hub contention, wrong driver).

## Capture loop

`synthmimic/capture_loop.py`:

```python
import hashlib, time, logging
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy.stats.qmc import Sobol

log = logging.getLogger(__name__)

class CaptureLoop:
    def __init__(self, profile, sender, capturer, out_dir, latency_ms=0):
        self.profile = profile
        self.sender = sender
        self.capturer = capturer
        self.out_dir = Path(out_dir)
        self.wav_dir = self.out_dir / "wav"
        self.wav_dir.mkdir(parents=True, exist_ok=True)
        self.latency_ms = latency_ms
        self.stats = {"captured": 0, "silent": 0, "retried": 0, "failed": 0}

    def reset(self):
        for msg in self.profile["reset"]["messages"]:
            t = msg["type"]
            if t == "cc":
                self.sender.send_cc(msg["cc"], msg["value"])
            elif t == "cc_14bit":
                self.sender.send_cc_14bit(msg["cc_msb"], msg["cc_lsb"], msg["value"])
            elif t == "nrpn":
                self.sender.send_nrpn(msg["nrpn_msb"], msg["nrpn_lsb"], msg["value"])
            elif t == "sysex":
                self.sender.send_sysex(msg["hex"])
            elif t == "sysex_file":
                self.sender.send_sysex_file(msg["path"])
            elif t == "sleep":
                time.sleep(msg["ms"] / 1000)
        settle = self.profile["reset"].get("settle_ms", 20)
        time.sleep(settle / 1000)

    def probe_one(self, params_dict, note):
        """One (params, note) capture. Returns audio array, or None on failure."""
        from .midi_sender import apply_params  # or move apply_params into sender
        self.reset()
        apply_params(self.sender, params_dict, self.profile)

        hold_sec = self.profile["probe"]["hold_ms"] / 1000
        rel_sec = self.profile["probe"]["release_ms"] / 1000
        total_sec = hold_sec + rel_sec + 0.3
        vel = self.profile["probe"]["velocity"]

        import threading
        audio_buf = []
        def _cap():
            audio_buf.append(self.capturer.capture(total_sec))
        t = threading.Thread(target=_cap); t.start()
        time.sleep(0.05)            # allow stream open
        self.sender.note_on(note, vel)
        time.sleep(hold_sec)
        self.sender.note_off(note)
        t.join()

        audio = audio_buf[0]
        # compensate measured latency by trimming
        trim = int(self.latency_ms * 48000 / 1000)
        audio = audio[trim:] if trim > 0 else audio
        return audio

    def is_healthy(self, audio):
        rms_db = 20 * np.log10(np.sqrt((audio ** 2).mean()) + 1e-9)
        return rms_db > self.profile["audio"]["silence_floor_db"]

    def run(self, n_samples, seed=0):
        modulated = [n for n, s in self.profile["parameters"].items()
                     if s.get("importance", 0) > 0]
        sobol = Sobol(d=len(modulated), scramble=True, seed=seed)
        vectors = sobol.random(n_samples)
        notes = self.profile["probe"]["notes"]

        rows = []
        for i, vec in enumerate(vectors):
            params_dict = dict(zip(modulated, vec))
            for note in notes:
                for attempt in range(3):
                    try:
                        audio = self.probe_one(params_dict, note)
                        if not self.is_healthy(audio):
                            self.stats["silent"] += 1
                            if attempt < 2:
                                self.stats["retried"] += 1
                                log.warning(f"Silent capture, retry {attempt+1}/2")
                                continue
                        h = hashlib.md5(vec.tobytes() + bytes([note])).hexdigest()[:12]
                        wav_path = self.wav_dir / f"{h}_n{note}.wav"
                        sf.write(wav_path, audio, 48000)
                        rows.append({
                            "hash": h, "note": note, "wav": str(wav_path),
                            **{f"p_{k}": v for k, v in params_dict.items()}
                        })
                        self.stats["captured"] += 1
                        break
                    except Exception as e:
                        log.error(f"Capture failed: {e}")
                        self.stats["failed"] += 1
                        time.sleep(1)
            if i % 100 == 0:
                log.info(f"Progress: {i}/{n_samples}  stats: {self.stats}")

        import pandas as pd
        pd.DataFrame(rows).to_parquet(self.out_dir / "samples.parquet")
```

## Entry point

`s02_capture/capture_v2.py` (entry point):

```python
import click, yaml, logging
from rich.logging import RichHandler
from synthmimic.profile import load_profile
from synthmimic.midi_sender import MidiSender
from synthmimic.audio_capturer import AudioCapturer
from synthmimic.capture_loop import CaptureLoop

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])

@click.command()
@click.option("--profile", required=True, help="Path to synth profile YAML")
@click.option("--session", default="config/session.yaml")
@click.option("-n", "--n-samples", default=10_000, type=int)
@click.option("--out-dir", default="data")
@click.option("--seed", default=0, type=int)
def main(profile, session, n_samples, out_dir, seed):
    prof = load_profile(profile)
    sess = yaml.safe_load(open(session))
    sender = MidiSender(sess["midi_port"], channel=prof["synth"]["channel"])
    cap = AudioCapturer(sess["audio_device"], sample_rate=48000,
                        block_size=sess.get("block_size", 256))
    loop = CaptureLoop(prof, sender, cap, out_dir,
                       latency_ms=sess.get("measured_latency_ms", 0))
    loop.run(n_samples, seed=seed)

if __name__ == "__main__":
    main()
```

Run:

```bash
# First: calibrate
python s02_capture/calibrate_latency.py

# Then capture
python s02_capture/capture_v2.py --profile s01_profiles/juno_106.yaml -n 50000
```

## Health checks & monitoring

The loop's `self.stats` is your monitoring dashboard. Rules of thumb:

| Metric                | Healthy   | Investigate                       |
| --------------------- | --------- | --------------------------------- |
| silent / captured     | < 2%      | Parameter mapping wrong, gain too low, synth stuck |
| retried / captured    | < 5%      | Reset protocol not settling       |
| failed / attempted    | 0         | MIDI port flaking, ASIO buffer underrun |
| Wall-clock per sample | ~3 s      | > 5s means reset is too slow      |

Log every 100 captures. For long runs, pipe `rich` output to a file + tail it from another terminal.

## Error modes and recovery

- **MIDI port disappears** (USB hub flake): catch `IOError`, reopen port, retry.
- **ASIO buffer underrun:** sounddevice raises `PortAudioError`. Log, skip this capture, continue.
- **Synth stuck note:** if the next capture is unusually loud, send All-Notes-Off + CC 120 (All Sound Off) before the reset.
- **Temperature drift (analog synths):** re-run latency calibration every ~1 hour; optionally re-run the synth's auto-tune.
- **Storage full:** check disk space in the health metric; abort early if < 10% free.

## Cross-platform notes

| Platform | Low-latency backend                | Recommended MIDI backend        |
| -------- | ---------------------------------- | ------------------------------- |
| Windows  | ASIO (vendor driver, or ASIO4ALL)  | `python-rtmidi` (mido default)  |
| macOS    | Core Audio (built-in)              | `python-rtmidi`                 |
| Linux    | JACK (or ALSA direct)              | `python-rtmidi`                 |

The code above works on all three; the only config change is `audio_device` in `session.yaml`.

## VST backend too (bonus)

Since V2's profile abstraction includes `encoding: vst`, you can run V2 against OB-Xf (or any other VST3) as well — the `MidiSender` is bypassed, and `apply_params` routes to a `VstSender` that uses DawDreamer. Useful for regression-testing V2 against V1's results on the same profile.

## When V2 is done

You're ready for Bucket 3 (dataset curation) when:

- A full run on the target synth completes without manual intervention.
- Silent / retry / failure rates are in the "healthy" ranges above.
- Latency calibration is stable within ±0.5 ms across sessions.
- Random spot-checks of captured WAVs reveal no artefacts (silence, clipping, stuck notes, contamination from previous patch).

## References

- [mido docs](https://mido.readthedocs.io/) · [python-rtmidi](https://spotlightkid.github.io/python-rtmidi/)
- [sounddevice (PortAudio wrapper)](https://python-sounddevice.readthedocs.io/)
- [PortAudio ASIO guide](https://files.portaudio.com/docs/v19-doxydocs/compile_windows_asio_msvc.html)
- [ASIO SDK (Steinberg)](https://www.steinberg.net/developers/) — background reading only, not needed at runtime.
- [[01 Profile - Juno 106]] · [[01 Profile - Novation Peak]] · [[01 Profile - OB-Xf]]
- [[Synth-Mimic-Pipeline]] · [[Hardware-in-the-Loop Capture]]
