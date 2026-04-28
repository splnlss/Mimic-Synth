---
tags: [build, 01-profile-obxf, synth-profile, ob-xf, vst, software, cross-platform]
created: 2026-04-19
synth: Surge Synthesizer Team OB-Xf
---

# 01 Profile — Surge Synth Team OB-Xf

> [!info] Quick facts
> **Type:** VST3 / AU / CLAP plugin (software, free, GPL-3) · **Modeled on:** Oberheim OB-X / OB-Xa lineage (continued development of OB-Xd) · **Voices:** 8 · **Source:** [github.com/surge-synthesizer/OB-Xf](https://github.com/surge-synthesizer/OB-Xf) · **Latest release:** v1.0.3

## Why OB-Xf as the V1 target

OB-Xf is the perfect prototyping target for [[02 Capture - VST]]:

- **Free, open-source, actively maintained.** The Surge Synthesizer team picked up the OB-Xd lineage and modernised it on JUCE 8.
- **Cross-platform.** Universal macOS binary (Intel + Apple Silicon), Windows x64, Linux x64/aarch64 — V1 runs identically on Mac and PC.
- **Three plug-in formats.** VST3, AU (macOS), and CLAP. DawDreamer hosts VST3 cleanly on all three OSes; AU is there as a fallback on macOS.
- **Open parameter set.** Every parameter is host-automatable with a known name and `[0.0, 1.0]` range. Param names are enumerable at runtime via the host.
- **No hardware round-trip.** Renders faster than real time, so a ~100k-sample dataset takes an hour, not a weekend.
- **Analog-style architecture.** Results generalise to the [[01 Profile - Juno 106|Juno]], [[01 Profile - Novation Peak|Peak]], and other subtractive synths.
- **Deterministic.** No voice drift, no analog aging, no warmup period — captures are bit-exact reproducible.

## Install paths (cross-platform)

Grab the latest release from [github.com/surge-synthesizer/OB-Xf/releases](https://github.com/surge-synthesizer/OB-Xf/releases) and install with the OS-appropriate installer.

| OS         | Format | Default path                                            |
| ---------- | ------ | ------------------------------------------------------- |
| macOS      | VST3   | `/Library/Audio/Plug-Ins/VST3/OB-Xf.vst3`               |
| macOS      | AU     | `/Library/Audio/Plug-Ins/Components/OB-Xf.component`    |
| macOS      | CLAP   | `/Library/Audio/Plug-Ins/CLAP/OB-Xf.clap`               |
| Windows    | VST3   | `C:\Program Files\Common Files\VST3\OB-Xf.vst3`         |
| Windows    | CLAP   | `C:\Program Files\Common Files\CLAP\OB-Xf.clap`         |
| Linux      | VST3   | `~/.vst3/OB-Xf.vst3` or `/usr/lib/vst3/OB-Xf.vst3`      |
| Linux      | CLAP   | `~/.clap/OB-Xf.clap` or `/usr/lib/clap/OB-Xf.clap`      |

For V1 we standardise on **VST3** because it's the most portable across DawDreamer / Pedalboard and the most consistent for parameter introspection.

## Parameter access: VST automation, not MIDI

As with OB-Xd, the primary control surface is the **VST parameter index**, not MIDI CC. DawDreamer exposes them by name:

```python
synth = engine.make_plugin_processor("obxf", "/path/to/OB-Xf.vst3")
for i in range(synth.get_plugin_parameter_size()):
    print(i, synth.get_parameter_name(i))
```

> [!warning] Always enumerate at runtime
> OB-Xf's parameter names are mostly inherited from OB-Xd but the Surge team has added, renamed, and reorganised a few (e.g. expanded filter types, additional voice options). Don't trust hard-coded names from older OB-Xd docs — generate the live list above and reconcile against the table in this profile before any dataset run. The reconciliation step belongs in your profile bootstrap script.

## Parameter map (representative — confirm at runtime)

> [!warning] Names below are illustrative — the live plugin uses spaced names like `"Osc 1 Pitch"`, `"Filter Cutoff"` that differ from the camelCase shortcuts shown here. Run the enumerator (`enumerate_params.py`) first, then reconcile against the table. The actual profile YAML (`s01_profiles/obxf.yaml`) and `reset:` block must use the exact strings the plugin exposes.

OB-Xf has ~60+ automatable parameters. The ones that matter most for [[Realtime Timbre Tracking]]:

| VST Param Name (typical) | Range         | Type       | Notes                       |
| ------------------------ | ------------- | ---------- | --------------------------- |
| Osc1Pitch                | 0.0 – 1.0     | continuous | coarse                      |
| Osc2Detune               | 0.0 – 1.0     | continuous |                             |
| OscMix                   | 0.0 – 1.0     | continuous |                             |
| Osc1Saw / Osc1Pulse      | 0.0 / 1.0     | enum       | per-osc waveform toggle     |
| PulseWidth               | 0.0 – 1.0     | continuous |                             |
| Xmod                     | 0.0 – 1.0     | continuous | cross-mod                   |
| Cutoff                   | 0.0 – 1.0     | continuous | ⭐                          |
| Resonance                | 0.0 – 1.0     | continuous | ⭐                          |
| FilterEnvAmt             | 0.0 – 1.0     | continuous |                             |
| Multimode                | 0.0 – 1.0     | continuous | LP → BP → HP morph          |
| FilterMode / HQ          | enum          | enum       | 12 vs 24 dB; OB-Xf may expose more types |
| Attack (VCA env)         | 0.0 – 1.0     | continuous |                             |
| Decay                    | 0.0 – 1.0     | continuous |                             |
| Sustain                  | 0.0 – 1.0     | continuous |                             |
| Release                  | 0.0 – 1.0     | continuous |                             |
| FilterAttack             | 0.0 – 1.0     | continuous |                             |
| FilterDecay              | 0.0 – 1.0     | continuous |                             |
| FilterSustain            | 0.0 – 1.0     | continuous |                             |
| FilterRelease            | 0.0 – 1.0     | continuous |                             |
| LfoFrequency             | 0.0 – 1.0     | continuous |                             |
| LfoAmount1               | 0.0 – 1.0     | continuous | → pitch mod                 |
| LfoAmount2               | 0.0 – 1.0     | continuous | → filter mod                |
| VoiceCount               | enum          | enum       | fix at 8 for training       |
| VoiceDetune              | 0.0 – 1.0     | continuous | analog-style drift          |
| Legato / Portamento      | various       | mostly freeze for v1        |

Every OB-Xf parameter is normalised to `[0.0, 1.0]` by the VST host. The profile stores the normalised range.

## Reset protocol

No MIDI involved — reset by writing canonical values to each VST parameter before every render:

```python
def reset(synth, profile, name_idx):
    """name_idx: {param_name: vst_index} dict built once via get_plugin_parameter_size()."""
    for name, canonical in profile["reset"].items():
        if name in name_idx:
            synth.set_parameter(name_idx[name], float(canonical))
```

Canonical init values (sensible midpoints — adjust after parameter-name reconciliation):

```yaml
reset:
  Osc1Pitch: 0.5
  Osc2Detune: 0.5
  OscMix: 0.5
  Osc1Saw: 1.0          # saw on
  Osc1Pulse: 0.0        # pulse off
  PulseWidth: 0.5
  Xmod: 0.0
  Cutoff: 0.7
  Resonance: 0.2
  FilterEnvAmt: 0.3
  Multimode: 0.0        # low-pass
  HQ: 1.0               # 24 dB (if present)
  Attack: 0.0
  Decay: 0.3
  Sustain: 1.0
  Release: 0.2
  FilterAttack: 0.0
  FilterDecay: 0.3
  FilterSustain: 0.8
  FilterRelease: 0.2
  LfoFrequency: 0.3
  LfoAmount1: 0.0
  LfoAmount2: 0.0
  VoiceCount: 1.0       # normalised = max voices
  VoiceDetune: 0.1
```

No settle delay needed — VST param changes are atomic.

## Recommended v1 subset

15 continuous params. Freeze everything else.

**Modulate live:**
Osc1Pitch, Osc2Detune, OscMix, PulseWidth, Xmod, Cutoff, Resonance, FilterEnvAmt, Multimode, Attack, Decay, Release, LfoFrequency, LfoAmount1, LfoAmount2.

**Freeze at defaults:** waveform toggles, HQ/filter type, VoiceCount = max, all per-filter-env params held at canonical values.

## Probe protocol

```yaml
probe:
  notes: [36, 48, 60, 72, 84]
  velocity: 100
  hold_sec: 1.5
  release_sec: 1.0
  sample_rate: 48000
  channels: 1    # mono render; OB-Xf is stereo but mono captures save space
  render_sec: 2.5
```

## Profile YAML snippet

```yaml
synth:
  id: obxf
  name: "Surge Synth Team OB-Xf"
  version: "1.0.3"
  transport: vst_host
  # Pick the path for your platform; profile bootstrap can autodetect.
  plugin_path_macos:   "/Library/Audio/Plug-Ins/VST3/OB-Xf.vst3"
  plugin_path_windows: "C:/Program Files/Common Files/VST3/OB-Xf.vst3"
  plugin_path_linux:   "~/.vst3/OB-Xf.vst3"
  parameter_encoding: vst_automation

parameters:
  Cutoff:
    encoding: vst
    range: [0.0, 1.0]
    continuous: true
    log_scale: true
    importance: 1.0
  Resonance:
    encoding: vst
    range: [0.0, 1.0]
    continuous: true
    importance: 0.95
  FilterEnvAmt:
    encoding: vst
    range: [0.0, 1.0]
    continuous: true
    importance: 0.85
  # ... (complete after runtime parameter enumeration)

probe:
  notes: [36, 48, 60, 72, 84]
  velocity: 100
  hold_sec: 1.5
  release_sec: 1.0
  render_sec: 2.5
  sample_rate: 48000

reset:
  # (see Reset protocol section)
```

## Quirks

- **Voice allocation is deterministic when VoiceDetune = 0.** Keep it near zero for clean training; add variation later.
- **Oscillator phase is not reset between DawDreamer render calls.** Waveforms for identical parameters differ across sequential renders (confirmed in v1.0.3 — startup logs "Suppressing default patch until we sort out race condition"). Spectral centroid CV is ~2–3% across renders, so timbral embeddings are stable but waveform-level loss is not. Use spectral/embedding losses, not raw MSE.
- **Portamento carries between renders.** Turn it off in the init, or you'll capture artefacts of the previous note.
- **Different builds may differ.** Lock to a specific OB-Xf release tag (e.g. `v1.0.3`) in the profile metadata. Param names occasionally shift across releases.
- **Apple Silicon vs Intel rendering.** Both produce identical samples in our testing, but pin the architecture in dataset metadata if you mix machines.
- **CLAP support is real but newer.** If DawDreamer's CLAP host has issues on your platform, fall back to VST3 — it's the most portable path.

## Migrating from OB-Xd

If you have existing OB-Xd captures or profile YAML, the migration is mostly mechanical:

1. Install OB-Xf alongside OB-Xd (different bundle ID; they coexist cleanly).
2. Run the parameter enumerator (snippet above) and diff against your OB-Xd parameter list. Most names match; reconcile any renames into the profile YAML.
3. Re-render a small calibration set (~100 captures) and confirm the spectral shape vs. cutoff sweep matches qualitatively. Don't expect bit-exact agreement — the underlying DSP has been refined.
4. Discard old OB-Xd surrogate weights; OB-Xf needs its own training pass. The pipeline architecture is unchanged.

## References

- [OB-Xf releases on GitHub](https://github.com/surge-synthesizer/OB-Xf/releases) — binaries for macOS, Windows, Linux.
- [OB-Xf source repo](https://github.com/surge-synthesizer/OB-Xf) — JUCE 8, GPL-3.
- [Original OB-Xd by discoDSP](https://www.discodsp.com/obxd/) — for historical context.
- [DawDreamer docs](https://github.com/DBraun/DawDreamer) — VST3 hosting from Python on all three OSes.
- [[Synth-Mimic-Pipeline]] · [[02 Capture - VST]]
