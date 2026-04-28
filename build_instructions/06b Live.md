---
tags: [build, 06b-live, streaming, midi, performance]
created: 2026-04-19
updated: 2026-04-19
---

# 06b Live — Continuous-Control Path

> [!info] Goal
> The continuous path: rolling audio frame in, smoothed MIDI CC stream out. This is the "talking synth" — a modulator that tracks a live source's timbre, not a one-shot patch generator. **Latency is a soft constraint, not a hard one** — aim for "feels responsive" (somewhere in the 100–300 ms range is plenty for the target use cases), not "sub-30 ms stage-ready." That relaxation simplifies several downstream choices.

Bucket 6b is where the whole project stops being an offline research exercise and becomes a playable instrument — but "playable" here means tracking an animal call, a voice, or an instrument phrase over seconds, not triggering percussive hits in lockstep. It reuses the surrogate and inverse from [[05 Surrogate]] / [[06 Invert]], but swaps the offline batch-CMA path for a streaming inference path running at a relaxed control rate.

> [!note] 2026-04-19 — latency target relaxed
> Earlier drafts specified a <30 ms end-to-end target. That forced a lot of architectural compromises (distilled CLAP, tiny context windows, 100 Hz CC output, DIN MIDI bandwidth panic). For the target use cases — animal-call mimicry, timbre tracking over phrases — 100–300 ms is indistinguishable from "instant" perceptually. This revision takes advantage of the slack: same EnCodec embedder as training (no distillation), larger context windows for stability, lower CC rates, and no bandwidth anxiety.

## Where Bucket 6b sits

```
live audio source ──► audio interface ──► rolling frame buffer
                                                 │
    ┌────────────────────────────────────────────┤
    ▼                                            │
pitch tracker  (CREPE-tiny / pYIN)               │
    │                                            ▼
    │                                   EnCodec encoder
    │                                   (same model as Bucket 4 / 5)
    │                                            │
    ▼                                            ▼
loudness follower              inverse model g(emb_window → params)
    │                                            │
    └─────────────┬──────────────────────────────┘
                  ▼
         per-CC slew / LPF
                  │
                  ▼
            MIDI output
       (CC, NRPN, 14-bit CC
        scheduled per profile)
                  │
                  ▼
              the synth
```

Three "control streams" merge at MIDI out:

1. **Pitch & note-on/off** — from the pitch tracker (deterministic DSP).
2. **Loudness / velocity / CC7** — from an RMS or BS.1770 follower (deterministic DSP).
3. **Timbral CCs** — from the neural inverse model (this is the learned part).

Keeping these streams separate means the pitch and amplitude are always correct, regardless of whether the neural model is having a bad day. The learned part only controls the CCs that carry timbre.

## Prerequisites

- [[05 Surrogate|Bucket 5]] static surrogate working and validated.
- [[06 Invert|Bucket 6]] offline inversion working — gives you confidence the mapping is learnable.
- A trained **inverse model** suitable for streaming (see "Training the inverse" below).
- An audio interface that can capture at 48 kHz and send MIDI out. No tight round-trip requirement — a stock USB interface with ~20–50 ms round-trip is fine.

> [!warning] Dataset gap vs Bucket 3
> [[03 Dataset|Bucket 3]] produces single-frame `(param, note, audio)` captures. Training a streaming inverse well requires **time-aligned sequences** — short param sweeps synchronised with the audio they produced. That means a small additional capture pass (call it "Bucket 3.5" or an addendum to Bucket 3) that renders ~5–10 s param-automation trajectories rather than held notes. Without this data the streaming inverse will lack temporal coherence and will jitter. See "Training the inverse" below.

## Latency budget (relaxed)

Target end-to-end latency: **~100–300 ms**, with responsive-feeling inner-loop updates. Split this into components so each has a budget, but without the sub-30 ms pressure the earlier draft assumed:

| Stage                                    | Budget (ms) | Notes |
| ---------------------------------------- | ----------- | ----- |
| Audio input buffering                    | 10–25       | 512–1024 samples at 48 kHz; comfortable, no driver tuning |
| EnCodec encoder (per frame)              | 3–8         | CPU; no distillation needed |
| Context window aggregation (N frames)    | 50–200      | Averaging over the last 4–16 EnCodec frames at 75 Hz |
| Pitch tracker (CREPE / pYIN)             | 10–40       | Parallel to embedding, not serial |
| Inverse model forward pass               | 1–5         | Small CNN+MLP; negligible on CPU |
| Per-CC smoothing / history buffer        | <1          | Trivial |
| MIDI output scheduling + transport       | 1–5         | USB MIDI; DIN is fine at our relaxed CC rate |
| Synth response (voicing + filter time)   | 5–20        | Synth-side; budget for it, can't reduce it |
| **Total**                                | **~100–300** | |

Why the change: a 20–30 ms target forced distillation, forced 100 Hz CC output, and made DIN MIDI bandwidth a near-miss. Loosening to ~200 ms eliminates all three constraints at effectively zero perceptual cost for the animal-call / voice / instrument-phrase targets this project is built around. A bird call lasts 500–2000 ms; 200 ms of tracking lag is inaudible as lag.

Tradeoff to be honest about: percussive, attack-critical use cases (drum triggers, live-looping beat matching) do not fit in this budget. If those become a requirement later, the whole architecture shifts back toward the tight-latency design (distillation, smaller buffers, 100 Hz CCs). Flag before scope-creeping in that direction.

## Streaming embedder

**Use EnCodec — the exact same 48 kHz model as [[04 Embed|Bucket 4]].** No distillation, no student network, no separate live embedder.

At ~3 ms/frame on CPU, EnCodec is well inside the relaxed latency budget even without GPU. The big architectural win: the surrogate in Bucket 5 was trained against EnCodec latents; the inverse model in this bucket is trained to map EnCodec latents → params; the live path encodes incoming audio with EnCodec. One embedder, end-to-end, no alignment step, no risk of training/serving skew.

Two variants of the same encoder depending on how much temporal smoothing you want:

- **Per-frame (75 Hz):** the raw EnCodec latent sequence, fed straight into the inverse with a context window.
- **Window-averaged (5–25 Hz effective):** average the last N frames before feeding the inverse. Trades responsiveness for stability. The relaxed latency budget makes this the default — more frames averaged → more stable CCs → less audible jitter on the synth.

Only reach for a smaller custom embedder if EnCodec inference becomes the bottleneck on the deploy target (rare at ~3 ms/frame). If it does, a 4-layer mel-CNN trained against EnCodec latents (distilled from them) is the escape hatch — but that's an optimisation, not a prerequisite.

## Training the inverse for streaming use

The static inverse from [[06 Invert|Bucket 6]] maps a single embedding → a single parameter vector. The streaming inverse needs two extra properties:

- **Temporal context.** Input is a window of the last N frames of EnCodec embeddings, not just the current frame. With the relaxed latency budget, N can be larger than the original sketch — typical N = 8–24 frames at 75 Hz (~100–320 ms of context). More context → more stable predictions and easier training, at the cost of a longer warm-up before the inverse starts emitting useful output.
- **Smoothness regularisation.** Loss includes a term penalising large frame-to-frame jumps in predicted params.

```python
# s06b_live/streaming_inverse.py — sketch
import torch
import torch.nn as nn

class StreamingInverse(nn.Module):
    def __init__(self, d_embed=128, d_params=15, context=16, hidden=256):
        """d_embed=128 matches EnCodec's continuous-latent dim (Bucket 4).
        context=16 frames at 75 Hz ≈ 213 ms — fine under the relaxed latency budget."""
        super().__init__()
        self.context = context
        # 1D conv over the context window, then an MLP head
        self.conv = nn.Sequential(
            nn.Conv1d(d_embed, hidden, kernel_size=3, padding=1), nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1), nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * context, hidden), nn.SiLU(),
            nn.Linear(hidden, d_params), nn.Sigmoid(),
        )

    def forward(self, emb_window):
        # emb_window: [B, context, d_embed]
        x = self.conv(emb_window.transpose(1, 2))        # [B, hidden, context]
        x = x.flatten(1)
        return self.head(x)                              # [B, d_params]

def smoothness_loss(params_seq):
    # params_seq: [B, T, d_params]
    return ((params_seq[:, 1:] - params_seq[:, :-1]) ** 2).mean()
```

Training loss: `mse(pred_params, true_params) + 0.2 * smoothness_loss(pred_params_seq)`. The smoothness weight goes up slightly compared to the tight-latency design — with a longer averaging window we can afford to over-smooth and recover responsiveness via per-CC time constants below.

**Where does the sequence training data come from?** Two paths — both involve an extra pass on the capture rig:

1. **Interpolated Sobol sequences.** Pick pairs of Sobol vectors and linearly interpolate over 5–10 seconds at 50–100 Hz, driving the live embedding model over the resulting audio. Cheap, synthetic, gives wide coverage.
2. **Recorded performer trajectories.** A human turns knobs on the synth's front panel (or a control surface) while audio records; the MIDI controller's CC output is captured as ground-truth params. More musical, more realistic, fewer samples.

Use (1) as the dataset and (2) as the eval set. Both emit the same schema: `(param_trajectory[T, d_params], audio[T * hop])`.

## The live loop (deployable pseudocode)

```python
# s06b_live/live.py — pseudocode
embedder       = Embedder()            # Bucket 4's EnCodec wrapper, no distillation
inverse        = StreamingInverse(context=16).eval()
pitch_tracker  = CrepeTiny()
loudness       = BS1770Follower()
smoother       = PerCCSmoother(taus_ms={"Cutoff": 40, "Resonance": 40, "OscMix": 150})
midi_out       = MidiSender(profile)

emb_history = deque(maxlen=16)          # ~213 ms at 75 Hz — matches context window

def on_audio_frame(frame, sr):
    emb = embedder.encodec_sequence(frame, sr)         # [128, T_frame]
    # flatten any T_frame > 1 into the history (frame length ~13 ms)
    for t in range(emb.shape[1]):
        emb_history.append(emb[:, t])

    if len(emb_history) < emb_history.maxlen:
        return                                         # warm-up

    window = torch.stack(list(emb_history)).unsqueeze(0)   # [1, 16, 128]
    with torch.no_grad():
        params = inverse(window).squeeze(0)            # [d_params]

    params_smoothed = smoother.update(params)

    # pitch + amplitude split
    pitch = pitch_tracker(frame, sr)                   # Hz or None
    rms   = loudness(frame)

    # note-on/off logic
    if rms < SILENCE_THRESH:
        midi_out.note_off_all()
    else:
        midi_out.note_on_if_changed(hz_to_midi(pitch), velocity=rms_to_vel(rms))
        midi_out.send_params(params_smoothed)          # the timbral CC stream
```

Deploy via [nn~](https://github.com/acids-ircam/nn_tilde) inside Max/Pd, or wrap as a [Neutone](https://github.com/Neutone/neutone_sdk) plug-in so it runs in any DAW.

## Per-CC smoothing

Different parameters have different perceptual speed. Hard filter-cutoff steps sound like zipper-noise; hard pan steps sound natural. Pick per-parameter time constants — values nudged slightly upward from the tight-latency draft, matching the relaxed context window:

| Parameter         | Time constant (ms) | Rationale |
| ----------------- | ------------------ | --------- |
| Filter cutoff     | 30–50              | Fast enough to feel connected to the source; slower than tight-budget design |
| Resonance         | 30–50              | Matches cutoff response |
| Filter drive      | 80                 | Slower; avoids artefacts near clipping |
| Osc mix           | 150                | Slow; otherwise crossfades audibly "step" |
| Envelope depths   | 50–100             | Per-note, not per-frame |
| LFO rate / depth  | 150+               | Subjectively slower effects |
| PWM               | 30                 | Fast when it matters |

One-pole IIR per parameter: `y[n] = α · x[n] + (1−α) · y[n−1]` with `α = 1 − exp(−1 / (τ · fs_ctrl))`. See [[Parameter Smoothing]] for the derivation.

## MIDI bandwidth (no longer a concern)

DIN MIDI 1.0 = 31.25 kbaud ≈ 1000 CC messages per second under ideal conditions. At the relaxed control rate of **10–25 Hz** (one CC update every 40–100 ms per parameter, which is all the ear can meaningfully react to anyway), we use a small fraction of the budget:

- **v1 live budget:** 8 expressive CCs at 20 Hz = 160 msgs/s. Trivial.
- **v2 live budget:** 12 CCs at 25 Hz + 14-bit CC pairs for cutoff/resonance = ~350 msgs/s. Still fine.

The earlier draft's 100 Hz CC target was downstream of the <30 ms latency goal. With the relaxed budget, a 20–25 Hz CC emit rate is plenty — the per-CC smoothers interpolate between updates so the synth never hears steps. DIN MIDI, USB MIDI, virtual MIDI buses — all fine.

See [[MIDI Bandwidth Budgeting]] for the full table and tradeoffs.

## UX modes

Bucket 6b supports two interaction modes from the same core loop:

- **Follow.** Stream params continuously while audio is present. The synth is always tracking the source. Best for duet-style performance with an instrument or voice.
- **Echo.** Capture N seconds of source audio + the generated param trajectory. Play it back on demand — source silences, synth echoes the captured timbre trajectory as a preset-plus-automation clip. Best for call-and-response with animal recordings.

Both modes share the same inverse model and smoother; echo mode just records the output stream to a file and replays it via the `MidiSender`.

## Validation

Before calling Bucket 6b done:

1. **Latency calibration.** Send a click into the audio input, measure the time from click to first non-reset CC change on MIDI out. Should land in the 100–300 ms range. Values well above that mean the context window is too long or the audio buffer is too large.
2. **Hold-test.** Play a sustained note into the input at a steady pitch and timbre. The CC output should converge to a stable value within ~300–500 ms and stay there (no jitter > ~1% of range).
3. **Sweep-test.** Slowly sweep a filter on the source over 3–5 seconds. The synth's filter should track it, with ~100–200 ms lag but no zipper and no oscillation.
4. **Silence gate.** Mute the source. The synth should send note-off and stop updating CCs within ~300 ms.
5. **Drift-free running.** Run the loop for 30 minutes on a varied source. MIDI throughput should stay well below the DIN budget (it will, trivially); no memory growth.

## Dependencies

```bash
pip install torch sounddevice soundfile numpy crepe  # or pyin via librosa
pip install encodec                                   # same embedder as Bucket 4
pip install mido python-rtmidi
# deployment
pip install neutone_sdk   # for plugin export
# Max / Pd side: install nn~ via https://github.com/acids-ircam/nn_tilde
```

## Uncertainties to flag

- **The sequence-dataset gap is the biggest open question.** [[03 Dataset|Bucket 3]] as currently specified produces single-frame captures. Without an extra "param-trajectory + audio" capture pass, the streaming inverse will not have the temporal coherence needed for stable use. Plan for a ~1–2 day Bucket 3 addendum before Bucket 6b's inverse training. The relaxed latency budget makes this *more* tractable, not less — longer trajectories, lower control rates, easier to collect.
- **"Relaxed latency" has a floor.** Even at 200 ms, audible lag exists for percussive or attack-heavy material. For the target use cases (animal mimicry, voice tracking, phrased instrument performance), this is fine. For drum/beat work it isn't — that would require rebuilding against the earlier tight-latency design. Know which camp your use case sits in before committing.
- **EnCodec shared across training and live removes the biggest prior risk.** The earlier "distil CLAP" open question is gone. The cost is slightly slower per-frame inference than a distilled student would have provided — acceptable given the relaxed budget.
- **Pitch tracker choice depends on target.** CREPE-tiny is reliable on voice and monophonic instruments; pYIN is better on noisy / broadband sources (bird calls often are). Pick per use case; don't assume one size fits all. With the relaxed budget, full CREPE (not tiny) is also an option.
- **MIDI 2.0 property exchange no longer matters.** Previous drafts flagged this as a bandwidth escape hatch. At our relaxed CC rate we never approach MIDI 1.0's limit, so MIDI 2.0 is an optional upgrade, not a solution to a real bottleneck.
- **Follow vs Echo mode trade time-alignment complexity.** Follow is best-effort streaming. Echo can be post-processed (re-run the inverse on the recorded audio offline for best-of-both). They share an inverse model but not a quality ceiling.
- **Neutone SDK platform limits.** Neutone wraps your model as an AU/VST3 that runs in DAWs, but it assumes a specific model interface and sample rate. Read the SDK docs before committing to it as your deployment path.
- **Context-window warm-up is a real UX thing.** With context=16 at 75 Hz, the inverse emits nothing useful for the first ~213 ms after audio starts. A short UX cue (e.g. don't send note-on to the synth until context is full) avoids a distracting "wrong sound → right sound" transition at the start of each phrase.

## When Bucket 6b is done

- `on_audio_frame(frame, sr) -> MIDI stream` runs continuously without dropping frames for 30+ minutes.
- Measured end-to-end latency sits in the 100–300 ms range — responsive-feeling for phrase-level tracking, no audible stepping, no dropouts.
- All 5 validation tests pass.
- Deployable as either a standalone Python process or an nn~ patch inside Max / Pd (or a Neutone plug-in in a DAW).
- Follow and Echo modes both working.
- Uses the same EnCodec embedder as [[04 Embed|Bucket 4]] — no separate distilled model to maintain.

## References

- [nn~ / nn_tilde](https://github.com/acids-ircam/nn_tilde) · [Neutone SDK](https://github.com/Neutone/neutone_sdk)
- [CREPE](https://github.com/marl/crepe) · [pYIN (librosa)](https://librosa.org/doc/main/generated/librosa.pyin.html)
- [torchaudio streaming](https://pytorch.org/audio/stable/io.html)
- [BS.1770 / EBU R128 loudness](https://tech.ebu.ch/publications/r128)
- [RAVE (IRCAM ACIDS)](https://github.com/acids-ircam/RAVE) — streaming neural audio reference.
- [BRAVE](https://arxiv.org/abs/2402.15015) — closely related realtime timbre work.
- [[Realtime Timbre Tracking]] · [[MIDI Bandwidth Budgeting]] · [[Parameter Smoothing]]
- [[05 Surrogate]] · [[06 Invert]] · [[08 Package]]
