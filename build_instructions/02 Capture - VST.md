---
tags: [build, 02-capture-vst, capture-rig, vst, proof-of-concept, cross-platform]
created: 2026-04-19
---

# 02 Capture — VST Proof of Concept (OB-Xf + DawDreamer)

> [!info] Goal
> Get the whole [[Synth-Mimic-Pipeline]] running end-to-end on a **software** synth in under a day, so the architecture is validated before touching hardware. Tightly coupled to [[01 Profile - OB-Xf|OB-Xf]]; intentionally *not* synth-agnostic (that's V2's job). Runs identically on **macOS, Windows, and Linux** — OB-Xf is universal.

## What V1 is and isn't

**Is:**
- Fast iteration: renders faster than real-time.
- A validation harness for the Bucket 3 → 4 → 5 → 6 pipeline.
- Deterministic: same seed → same dataset.
- Cross-platform. Same `capture_v1.py` on macOS / Windows / Linux with only a path difference.

**Isn't:**
- Synth-agnostic. Hard-codes OB-Xf specifics for speed.
- Hardware-ready. No MIDI plumbing, no audio interface, no reset ritual.
- Production-quality. Minimal error handling, no health checks.

Once this runs cleanly and a tiny surrogate converges, move to [[02 Capture - Hardware]].

## Dependencies

```bash
# Python 3.10+
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install dawdreamer numpy scipy soundfile pyarrow pyyaml pandas tqdm
```

You also need OB-Xf installed locally. Download the OS-appropriate installer from [github.com/surge-synthesizer/OB-Xf/releases](https://github.com/surge-synthesizer/OB-Xf/releases) (v1.0.3 or later).

Default install paths:

| OS         | VST3 path                                                |
| ---------- | -------------------------------------------------------- |
| macOS      | `/Library/Audio/Plug-Ins/VST3/OB-Xf.vst3`                |
| Windows    | `C:\Program Files\Common Files\VST3\OB-Xf.vst3`          |
| Linux      | `~/.vst3/OB-Xf.vst3`                                     |

> [!tip] DawDreamer + VST3 is the portable combo
> DawDreamer hosts VST3 on all three OSes. OB-Xf also ships as AU (macOS) and CLAP (all three), but VST3 is the only format that's consistent across platforms without code changes.

## Project layout

```
MimicSynth/
├── s01_profiles/
│   └── obxf.yaml              # the 01 Profile
├── s02_capture/
│   ├── capture_v1.py          # the script below
│   └── data/
│       ├── samples.parquet    # (param, note, audio_path)
│       └── wav/               # per-sample WAV files
├── s03_dataset/               # sampling, quality, sequences modules
└── .venv/
```

## The minimal capture script

`capture_v1.py` — a complete working script. Reads the profile, picks the OS-appropriate plugin path, Sobol-samples the parameter space, renders audio for each probe note, saves WAVs and a Parquet index.

```python
"""
Bucket 2 V1: OB-Xf capture rig.
Renders (parameter vector, note) -> WAV via DawDreamer.
Runs on macOS, Windows, Linux.
"""
from pathlib import Path
import hashlib
import platform
import numpy as np
import pandas as pd
import yaml
import soundfile as sf
from scipy.stats.qmc import Sobol
import dawdreamer as daw
from tqdm import tqdm

SAMPLE_RATE = 48000         # matches the EnCodec embedder (Bucket 4); no resample later
BUFFER_SIZE = 512
N_SAMPLES = 10_000          # start small; scale to 100k once working
PROFILE_PATH = "s01_profiles/obxf.yaml"
OUT_DIR = Path("data")
WAV_DIR = OUT_DIR / "wav"
WAV_DIR.mkdir(parents=True, exist_ok=True)


def load_profile(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_plugin_path(profile):
    """Pick the right plugin path for the current OS."""
    sys = platform.system()
    key = {
        "Darwin":  "plugin_path_macos",
        "Windows": "plugin_path_windows",
        "Linux":   "plugin_path_linux",
    }[sys]
    path = Path(profile["synth"][key]).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"OB-Xf not found at {path}. Install from "
            "https://github.com/surge-synthesizer/OB-Xf/releases"
        )
    return str(path)


def build_name_index(synth):
    """Return a {param_name: index} dict for the loaded plugin."""
    n = synth.get_plugin_parameter_size()
    return {synth.get_parameter_name(i): i for i in range(n)}


def apply_params(synth, params_dict, profile, name_idx):
    """Write normalised [0,1] param vector to the VST.
    Mutates params_dict in-place so stored values match what the synth received."""
    for name in list(params_dict):
        value = params_dict[name]
        spec = profile["parameters"][name]
        if not spec["continuous"]:
            n_cats = len(spec.get("categories", [])) or 2
            value = round(value * (n_cats - 1)) / (n_cats - 1)
            params_dict[name] = value  # keep stored value in sync with synth state
        synth.set_parameter(name_idx[name], float(value))


def reset(synth, profile, name_idx):
    for name, canonical in profile["reset"].items():
        if name in name_idx:
            synth.set_parameter(name_idx[name], float(canonical))


def render_one(engine, synth, note, profile):
    """Render a single (note, param-state) capture. Returns mono float32."""
    synth.clear_midi()
    vel = profile["probe"]["velocity"]
    synth.add_midi_note(note, vel, 0.0, profile["probe"]["hold_sec"])
    engine.set_bpm(120)
    engine.render(profile["probe"]["render_sec"])
    audio = engine.get_audio()               # (channels, samples)
    return audio.mean(axis=0).astype(np.float32)


def sample_vectors(n, modulated_params, seed=0):
    """Sobol-sample the modulated parameter subspace in [0,1]^d."""
    d = len(modulated_params)
    sobol = Sobol(d=d, scramble=True, seed=seed)
    return sobol.random(n)


def main():
    profile = load_profile(PROFILE_PATH)
    modulated = [
        name for name, spec in profile["parameters"].items()
        if spec.get("importance", 0) > 0
    ]
    notes = profile["probe"]["notes"]

    plugin_path = resolve_plugin_path(profile)
    engine = daw.RenderEngine(SAMPLE_RATE, BUFFER_SIZE)
    synth = engine.make_plugin_processor("obxf", plugin_path)

    # Sanity: enumerate parameters once so you can verify the profile matches.
    name_idx = build_name_index(synth)
    missing = [n for n in modulated if n not in name_idx]
    if missing:
        raise RuntimeError(
            f"Profile references parameters not exposed by this OB-Xf build: {missing}. "
            "Run the enumerator and reconcile profile YAML."
        )

    rows = []
    vectors = sample_vectors(N_SAMPLES, modulated)
    engine.load_graph([(synth, [])])  # load once; re-register only if graph changes

    for i, vec in enumerate(tqdm(vectors, desc="capturing")):
        reset(synth, profile, name_idx)
        params_dict = dict(zip(modulated, vec))
        apply_params(synth, params_dict, profile, name_idx)

        for note in notes:
            audio = render_one(engine, synth, note, profile)
            if np.max(np.abs(audio)) < 1e-4:
                print(f"  warning: silent render (vec={i}, note={note}) — skipping")
                continue
            h = hashlib.md5(vec.tobytes() + bytes([note])).hexdigest()[:12]
            wav_path = WAV_DIR / f"{h}_n{note}.wav"
            sf.write(wav_path, audio, SAMPLE_RATE)
            rows.append({
                "hash": h,
                "note": note,
                "wav": str(wav_path),
                **{f"p_{k}": v for k, v in params_dict.items()},
            })

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_DIR / "samples.parquet")
    print(f"Saved {len(df)} captures to {OUT_DIR}/samples.parquet")


if __name__ == "__main__":
    main()
```

Run:

```bash
python s02_capture/capture_v1.py
```

On a modern laptop (Apple Silicon or a recent x86): ~10k captures × 5 notes = 50k WAVs in 15–30 minutes. Storage: ~2.2 GB at 48 kHz/mono/2.5 s.

## Quick sanity check

Before training anything, verify the captures look right:

```python
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_parquet("s02_capture/data/samples.parquet")
# Pick 8 random captures with varied filter cutoff
s = df.sample(8).sort_values("p_Cutoff")
fig, ax = plt.subplots(len(s), 1, figsize=(10, 12))
for i, (_, r) in enumerate(s.iterrows()):
    audio, sr = sf.read(r["wav"])
    spec = np.abs(np.fft.rfft(audio))[:2000]
    ax[i].plot(20 * np.log10(spec + 1e-9))
    ax[i].set_title(f"Cutoff={r['p_Cutoff']:.2f} Note={r['note']}")
plt.tight_layout()
plt.show()
```

You should see the spectral rolloff shifting with Cutoff. If not, something's wrong with parameter application (or the parameter name doesn't match what OB-Xf actually calls it — run the enumerator and reconcile).

## Cross-platform notes

- **macOS (Intel & Apple Silicon):** install the universal `.pkg`. Gatekeeper may need to be reassured on first load (System Settings → Privacy & Security).
- **Windows 10/11 x64:** install via `.exe` or unzip `.vst3` into `C:\Program Files\Common Files\VST3\`. On some systems, DawDreamer benefits from adding the VST3 folder to `PATH` or using the full absolute path.
- **Linux (x64 / aarch64):** unzip `.vst3` into `~/.vst3/`. On Ubuntu, install `libasound2-dev` if DawDreamer complains about ALSA at import time.
- **Path separators:** the script resolves paths with `pathlib.Path`, so forward and back slashes both work — don't worry about escaping.

## What to validate next

1. **Embedding extraction** (Bucket 4): run CLAP over the WAVs, save embeddings alongside the Parquet. Confirm semantically similar captures cluster.
2. **Tiny surrogate** (Bucket 5): 3-layer MLP, 256 hidden, train for 20 epochs. Validation loss should drop to <0.1 cosine-distance.
3. **Inversion** (Bucket 6): pick a held-out capture, extract its embedding, search for the parameter vector that minimises surrogate-predicted embedding distance. Confirm recovered params are close to ground truth.

If steps 1–3 work on OB-Xf, the architecture is sound and you can confidently invest in V2.

## Known limitations (by design)

- OB-Xf only. No abstraction for other plugins or hardware.
- No active learning — pure Sobol.
- No frame-wise captures yet — single-vector embeddings only. [[Realtime Timbre Tracking]] needs sequential captures; add that in V2.
- Silent renders (e.g. stuck note, extreme envelope) are skipped with a warning, but the root cause is not diagnosed automatically.
- Uses mono downmix; stereo image discarded. Fine for v1.
- **OB-Xf oscillator phase is not reset between DawDreamer renders.** Waveforms for the same parameters differ across calls (up to ~0.19 peak amplitude), though spectral centroid stays within ~2–3%. This is a known OB-Xf v1.0.3 issue ("Suppressing default patch until we sort out race condition"). Timbral embeddings are stable; waveform-level loss functions are not.

## When to move to V2

Move to [[02 Capture - Hardware]] when:
- The full pipeline trains successfully on OB-Xf data and inverts known captures with reasonable accuracy.
- You're ready to target a specific hardware synth.
- You need the capture rig to be reusable across synths.

## References

- [OB-Xf releases on GitHub](https://github.com/surge-synthesizer/OB-Xf/releases) — universal installers, cross-platform.
- [DawDreamer on GitHub](https://github.com/DBraun/DawDreamer) — VST hosting, MIDI, rendering.
- [Pedalboard (Spotify)](https://github.com/spotify/pedalboard) — alternative VST host; swap if DawDreamer has issues on your platform.
- [[01 Profile - OB-Xf]] · [[Synth-Mimic-Pipeline]] · [[Quasi-Random Sampling]]
