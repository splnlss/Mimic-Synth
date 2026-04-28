---
tags: [build, 01-profile-peak, synth-profile, novation-peak, hardware]
created: 2026-04-19
synth: Novation Peak
---

# 01 Profile — Novation Peak

> [!info] Quick facts
> **Year:** 2017 · **Voices:** 8 · **Architecture:** Hybrid — NCO digital oscillators → analog filter + VCA · **MIDI:** USB MIDI + DIN 5-pin in/out/thru · **MPE:** supported · **Receive channel:** 1–16 or OMNI

## The good news

The Peak is a modern, well-behaved MIDI citizen. Unlike the [[01 Profile - Juno 106|Juno-106]], nearly every useful parameter is on a documented MIDI CC. NRPN is available for the remainder. The official "Novation Peak MIDI Implementation Chart" (PDF on Novation's support site) is exhaustive and accurate.

This means:
- **Parameter encoding is CC-first, NRPN-second, SysEx only for patch dumps.**
- You can use 14-bit CC (MSB/LSB pairs) for critical params where the 7-bit resolution would zipper.
- Device ID is 0x33. Manufacturer ID is 0x00 0x20 0x29 (Focusrite/Novation).

## Core CC map (abbreviated)

Full map is in the official chart — this is the subset most useful for v1.

| CC   | Parameter                    | Notes                      |
| ---- | ---------------------------- | -------------------------- |
| 1    | Mod Wheel                    |                            |
| 7    | Volume                       |                            |
| 11   | Expression                   |                            |
| 16   | Osc 1 Coarse                 |                            |
| 17   | Osc 1 Fine                   |                            |
| 18   | Osc 1 Wave Interp            | continuous wave morph      |
| 19   | Osc 1 Pulse Width            |                            |
| 20   | Osc 2 Coarse                 |                            |
| 22   | Osc 2 Wave Interp            |                            |
| 23   | Osc 2 Pulse Width            |                            |
| 29   | Osc 3 Wave Interp            |                            |
| 74   | Filter Frequency             | ⭐ (14-bit avail at CC 74 MSB + 42 LSB) |
| 71   | Filter Resonance             | ⭐                          |
| 72   | Env 1 Release                |                            |
| 73   | Env 1 Attack                 |                            |
| 75   | Env 1 Decay                  |                            |
| 76   | Env 1 Sustain                |                            |
| 77   | Filter Drive                 |                            |
| 82   | Filter Env Depth             |                            |
| 83   | Filter Type                  | enum (LP/BP/HP, slopes)    |
| 87   | Filter Key Tracking          |                            |
| 102  | LFO 1 Rate                   |                            |
| 103  | LFO 1 Sync                   | enum                       |
| 104  | LFO 1 Waveform               | enum                       |
| 105  | LFO 1 Depth                  |                            |

Full list: ~150+ CCs covering every panel control. Pull from the official chart.

## NRPN for mod matrix

The Peak's 16-slot mod matrix is controlled via NRPN. Each slot has:
- Source NRPN
- Destination NRPN
- Depth NRPN

NRPN format (standard):
```
CC 99 <MSB>    (NRPN parameter MSB)
CC 98 <LSB>    (NRPN parameter LSB)
CC 6  <value>  (data)
```

For v1 you can ignore the mod matrix entirely — fix all depths to zero in the reset, and let the surrogate learn the "no modulation" case only. Add mod matrix sampling in v2.

## Patch dump via SysEx

A single-patch dump/load is used in the reset protocol. The format is:
```
F0 00 20 29 00 33 <patch data> F7
```

Request current patch:
```
F0 00 20 29 00 33 40 F7
```

The cleanest reset is "load canonical init patch via SysEx" — package the init patch as a .syx and dispatch it once per sample. Much faster than sending dozens of individual CCs.

## Reset protocol

Two options:

**Option A — SysEx init dump (fast):**
```yaml
reset:
  messages:
    - {type: sysex_file, path: "s01_profiles/peak_init.syx"}
    - {type: cc, cc: 123, value: 0}  # All notes off
  settle_ms: 15
```

**Option B — CC sweep (slower, doesn't require a .syx):**
```yaml
reset:
  messages:
    # zero all mod depths
    - {type: cc, cc: 105, value: 0}  # LFO 1 depth
    # set base filter + envs
    - {type: cc_14bit, cc_msb: 74, cc_lsb: 42, value: 10000}  # Filter ~midpoint
    - {type: cc, cc: 71, value: 30}   # Resonance
    - {type: cc, cc: 73, value: 0}    # Attack
    - {type: cc, cc: 75, value: 64}   # Decay
    - {type: cc, cc: 76, value: 127}  # Sustain
    - {type: cc, cc: 72, value: 30}   # Release
    - {type: cc, cc: 7, value: 100}   # Volume
    - {type: cc, cc: 123, value: 0}   # All notes off
  settle_ms: 20
```

Option A is preferred once you have a canonical init patch saved.

## Recommended v1 parameter subset

15 continuous, 2 enumerated. Larger surface than the Juno because the Peak can handle it.

**Continuous (sample live):**
- Osc 1 Wave Interp (CC 18)
- Osc 1 Pulse Width (CC 19)
- Osc 2 Coarse (CC 20)
- Osc 2 Wave Interp (CC 22)
- Osc Mix (Peak has dedicated mix CCs per osc)
- Filter Frequency (CC 74, 14-bit) ⭐
- Filter Resonance (CC 71) ⭐
- Filter Drive (CC 77)
- Filter Env Depth (CC 82)
- Env 1 Attack (CC 73)
- Env 1 Decay (CC 75)
- Env 1 Sustain (CC 76)
- Env 1 Release (CC 72)
- LFO 1 Rate (CC 102)
- LFO 1 Depth (CC 105)

**Enumerated (treat categorically):**
- Filter Type (CC 83) — 6 values (LP12/LP24/BP6/BP12/HP12/HP24)
- LFO 1 Waveform (CC 104) — 8 values

## Probe protocol

```yaml
probe:
  notes: [36, 48, 60, 72, 84]   # C2–C6 (Peak handles wider range cleanly)
  velocity: 100
  hold_ms: 1200
  release_ms: 800
  channel: 1
```

## Known quirks

- **FPGA oscillators.** Digital oscillators feed an analog filter/VCA. Spectra are very precise, so the surrogate typically fits the Peak more accurately than an all-analog synth. Good news for benchmarking.
- **Voice drift off by default.** If you want "analog feel," enable voice drift in the init patch. For reproducible training captures, keep it off.
- **Firmware matters.** Different firmware revisions have slightly different CC maps (especially around the Animate buttons). Lock to one firmware version for a dataset run and record the version in the profile metadata.
- **USB MIDI recommended.** DIN works too but USB gives higher throughput and slightly lower jitter.
- **MPE on by default in some modes.** For single-timbre capture, set MPE off in the global settings.

## Profile YAML snippet

```yaml
synth:
  id: novation_peak
  name: "Novation Peak"
  firmware: "2.0.2"   # record exact firmware
  transport: usb_midi
  channel: 1
  parameter_encoding: cc
  supports_14bit_cc: true
  supports_mpe: true   # but disabled for training

parameters:
  filter_freq:
    encoding: cc_14bit
    cc_msb: 74
    cc_lsb: 42
    range: [0, 16383]
    continuous: true
    importance: 1.0
  filter_resonance:
    encoding: cc
    cc: 71
    range: [0, 127]
    continuous: true
    importance: 0.95
  osc1_wave_interp:
    encoding: cc
    cc: 18
    range: [0, 127]
    continuous: true
    importance: 0.8
  filter_type:
    encoding: cc
    cc: 83
    range: [0, 5]
    continuous: false
    categories: [LP12, LP24, BP6, BP12, HP12, HP24]
    importance: 0.6
  # ... (complete the map)

probe:
  notes: [36, 48, 60, 72, 84]
  velocity: 100
  hold_ms: 1200
  release_ms: 800

reset:
  messages:
    - {type: sysex_file, path: "s01_profiles/peak_init.syx"}
    - {type: cc, cc: 123, value: 0}
  settle_ms: 15
```

## References

- https://midi.guide/d/novation/summit-and-peak/ 
- [Sound on Sound Peak review](https://www.soundonsound.com/reviews/novation-peak) — for architecture context.
- [[Synth-Mimic-Pipeline]] · [[Hardware-in-the-Loop Capture]] · [[MIDI Implementation Charts]]
