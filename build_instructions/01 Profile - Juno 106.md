---
tags: [build, 01-profile-juno106, synth-profile, juno-106, hardware]
created: 2026-04-19
synth: Roland Juno-106
---

# 01 Profile — Roland Juno-106

> [!info] Quick facts
> **Year:** 1984 · **Voices:** 6 · **Architecture:** Analog subtractive (DCO → HPF → VCF → VCA) · **MIDI:** DIN 5-pin only (no USB) · **Receive channel:** 1–16 (set on rear panel)

## The important quirk: SysEx, not CC

The Juno-106 was one of the first synths with factory MIDI, and it shows. It has **extremely sparse MIDI CC support** but a **complete SysEx parameter interface**. Nearly every front-panel slider is addressable via SysEx.

- **CC works for:** mod wheel (CC1), volume (CC7), hold pedal (CC64). That's essentially it.
- **Everything timbral uses SysEx.** Address space is clean, message format is simple, every panel control is covered.

This shapes the whole profile — our parameter encoding is SysEx-first.

## SysEx message format

```
F0 41 32 0n pp vv F7
```

| Byte | Value | Meaning                               |
|------|-------|---------------------------------------|
| F0   | —     | SysEx start                           |
| 41   | —     | Roland manufacturer ID                |
| 32   | —     | Juno-106 parameter change ID          |
| 0n   | 0–F   | MIDI channel (0 = ch 1, F = ch 16)    |
| pp   | 00–19 | Parameter number                      |
| vv   | 00–7F | 7-bit value                           |
| F7   | —     | SysEx end                             |

Each message is 7 bytes → ~2.2 ms on DIN MIDI. Sending 12 parameters back-to-back = ~26 ms of MIDI alone. Factor that into the capture pacing in [[02 Capture - Hardware]].

## Parameter map

> [!warning] Verify against your unit
> Numbers below are from the standard Juno-106 MIDI Implementation Chart. SOPs and community-modded units can deviate. Send each message, listen for the expected change, confirm before building the full dataset.

| pp   | Parameter         | Range          | Type       |
| ---- | ----------------- | -------------- | ---------- |
| 0x00 | LFO Rate          | 0–127          | continuous |
| 0x01 | LFO Delay         | 0–127          | continuous |
| 0x02 | DCO LFO Mod       | 0–127          | continuous |
| 0x03 | DCO PWM Depth     | 0–127          | continuous |
| 0x04 | PWM Source        | 0 / 64 / 127   | enum (3)   |
| 0x05 | Noise Level       | 0–127          | continuous |
| 0x06 | Sub Osc Level     | 0–127          | continuous |
| 0x07 | Pulse Level       | 0–127          | continuous |
| 0x08 | Saw Level         | 0–127          | continuous |
| 0x09 | DCO Range         | 0 / 64 / 127   | enum (16'/8'/4') |
| 0x0A | HPF Cutoff        | 0 / 32 / 64 / 96 | enum (4) |
| 0x0B | VCF Cutoff        | 0–127          | continuous ⭐ |
| 0x0C | VCF Resonance     | 0–127          | continuous ⭐ |
| 0x0D | VCF Env Polarity  | 0 / 127        | enum (+/−) |
| 0x0E | VCF Env Depth     | 0–127          | continuous |
| 0x0F | VCF Kybd Follow   | 0 / 64 / 127   | enum (3)   |
| 0x10 | VCF LFO Mod       | 0–127          | continuous |
| 0x11 | VCA Mode          | 0 / 127        | enum (Env/Gate) |
| 0x12 | VCA Level         | 0–127          | continuous |
| 0x13 | Env Attack        | 0–127          | continuous |
| 0x14 | Env Decay         | 0–127          | continuous |
| 0x15 | Env Sustain       | 0–127          | continuous |
| 0x16 | Env Release       | 0–127          | continuous |
| 0x17 | Chorus Mode       | 0 / 43 / 85 / 127 | enum (Off/I/II/I+II) |

⭐ = highest perceptual weight; bias sampling toward these.

## Recommended v1 subset

Start with **12 continuous parameters** and freeze the enumerated ones at sensible defaults. This keeps Bucket 3's sampling space tractable and avoids the categorical-variable complications in [[Neural Surrogate for Synth]] v1.

**Modulate live:**
LFO Rate, DCO LFO Mod, DCO PWM Depth, Sub Osc Level, Pulse Level, Saw Level, VCF Cutoff, VCF Resonance, VCF Env Depth, Env Attack, Env Decay, Env Release.

**Freeze at:**
- DCO Range = 8' (64)
- HPF = 0
- VCF Env Polarity = + (0)
- VCF Kybd Follow = Half (64)
- VCA Mode = Env (0)
- Chorus = Off (0)

## Reset protocol

The Juno-106 has no "init patch" MIDI command. Reset is a scripted SysEx sequence that pushes every parameter to a canonical midpoint before each capture.

```yaml
reset:
  messages:
    - {type: sysex, hex: "F0 41 32 00 00 40 F7"}  # LFO rate = 64
    - {type: sysex, hex: "F0 41 32 00 01 00 F7"}  # LFO delay = 0
    - {type: sysex, hex: "F0 41 32 00 02 00 F7"}  # DCO LFO mod = 0
    - {type: sysex, hex: "F0 41 32 00 03 00 F7"}  # PWM depth = 0
    - {type: sysex, hex: "F0 41 32 00 05 00 F7"}  # Noise = 0
    - {type: sysex, hex: "F0 41 32 00 06 40 F7"}  # Sub = 64
    - {type: sysex, hex: "F0 41 32 00 07 40 F7"}  # Pulse = 64
    - {type: sysex, hex: "F0 41 32 00 08 40 F7"}  # Saw = 64
    - {type: sysex, hex: "F0 41 32 00 0B 60 F7"}  # VCF cutoff = 96
    - {type: sysex, hex: "F0 41 32 00 0C 20 F7"}  # Resonance = 32
    - {type: sysex, hex: "F0 41 32 00 0E 40 F7"}  # Env depth = 64
    - {type: sysex, hex: "F0 41 32 00 13 00 F7"}  # Attack = 0
    - {type: sysex, hex: "F0 41 32 00 14 40 F7"}  # Decay = 64
    - {type: sysex, hex: "F0 41 32 00 15 7F F7"}  # Sustain = 127
    - {type: sysex, hex: "F0 41 32 00 16 20 F7"}  # Release = 32
    # freeze enumerated:
    - {type: sysex, hex: "F0 41 32 00 09 40 F7"}  # Range = 8'
    - {type: sysex, hex: "F0 41 32 00 0A 00 F7"}  # HPF = 0
    - {type: sysex, hex: "F0 41 32 00 0D 00 F7"}  # Env pol = +
    - {type: sysex, hex: "F0 41 32 00 0F 40 F7"}  # Kybd follow = Half
    - {type: sysex, hex: "F0 41 32 00 11 00 F7"}  # VCA = Env
    - {type: sysex, hex: "F0 41 32 00 17 00 F7"}  # Chorus = Off
    - {type: cc, cc: 7, value: 100}                # Volume
    - {type: cc, cc: 123, value: 0}                # All notes off
  settle_ms: 30
```

## Probe protocol

```yaml
probe:
  notes: [36, 48, 60, 72]    # C2, C3, C4, C5
  velocity: 100
  hold_ms: 1500
  release_ms: 1000
  channel: 1
```

## Known quirks

- **Voice chip failure (80017A).** The voice chips degrade with age; silent or weak voices are common. Run a mono-voice diagnostic before a dataset run — play the same note with a known patch across all 6 voices and compare RMS. If one voice is >2 dB off, exclude it in the probe protocol (play only notes that round-robin away from it, or have the chip replaced).
- **DIN-only.** No USB MIDI. You need a USB-to-DIN interface for [[Hardware-in-the-Loop Capture]].
- **No program-change reset.** Always send the full SysEx reset.
- **Tuning drift.** Analog. Warm the unit for 20 minutes before starting a dataset run. Consider re-running the auto-tune periodically.
- **Chorus bleed.** Even with Chorus = Off, the Juno's audio path is slightly coloured. Not a problem — the surrogate learns it — but note that this synth's "dry" output is not textbook dry.

## Profile YAML snippet

```yaml
synth:
  id: roland_juno_106
  name: "Roland Juno-106"
  transport: din_midi
  channel: 1
  parameter_encoding: sysex

parameters:
  vcf_cutoff:
    encoding: sysex
    param_num: 0x0B
    range: [0, 127]
    continuous: true
    importance: 1.0
  vcf_resonance:
    encoding: sysex
    param_num: 0x0C
    range: [0, 127]
    continuous: true
    importance: 0.9
  env_attack:
    encoding: sysex
    param_num: 0x13
    range: [0, 127]
    continuous: true
    log_scale: true
    importance: 0.7
  # ... (complete the map from the table above)

probe:
  notes: [36, 48, 60, 72]
  velocity: 100
  hold_ms: 1500
  release_ms: 1000

reset:
  # (see Reset protocol section)

quirks:
  - voice_chip_diagnostic: required
  - warmup_minutes: 20
  - din_only: true
```

## References

- Roland Juno-106 Service Manual — MIDI Implementation Chart in the appendix.
- http://www.hinzen.de/midi/juno-106/howto-02.html
- [[Synth-Mimic-Pipeline]] · [[Hardware-in-the-Loop Capture]] · [[MIDI Implementation Charts]]
