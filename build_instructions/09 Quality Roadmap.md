# 09 Quality Roadmap — From Sparse Prototype to Convincing Mimicry

## Where we are

The current pipeline produces a recognisable approximation of the source — correct rhythm, correct pitch regions, some timbral overlap — but the result sounds like a basic synth patch rather than the source material. Three root causes:

1. **Sparse tonal synthesis.** The surrogate was trained on static single-note patches. It has no concept of timbral layering, harmonic density, noise injection, or the way a complex sound is built from many simultaneous synthesis decisions.
2. **Objective function blindness.** EnCodec embeddings + MRSTFT + aperiodicity are useful but none of them directly tell the optimizer "this harmonic relationship is wrong" or "the attack transient is missing spectral complexity." The scoring rewards global timbral proximity, not the perceptually critical details.
3. **No synthesis knowledge.** CMA-ES explores the parameter space statistically. It has no understanding of what makes a sound rich: layered oscillators at musical intervals, aggressive filter movement, noise mixed into the harmonic structure, or the interaction between amp and filter envelopes.

---

## I. Richer Synthesis — Oscillator and Signal Path

### A. Oscillator layering
The single-voice OB-Xf patch is the primary limitation. A convincing mimic of a complex organic sound needs at minimum:
- **Two oscillators at a musical interval** — unison (beating), fifth (+7 st), octave (+12 st), or minor third (+3 st). The current search treats Osc 2 Pitch as a floating parameter; it should be snapped to meaningful harmonic intervals (± integer semitones) as a discrete outer loop, the same way osc config scouting tries saw/pulse/saw+pulse.
- **Oscillator sync** — hard sync creates inharmonic tearing transients that read as "complex" to a listener. OB-Xf exposes this; the current pipeline ignores it entirely.
- **Sub-oscillator** — adding a square wave one octave below fills the low-mid register that pure saw/pulse waves lack.

### B. Noise as a first-class voice
The current treatment adds noise as a continuous background layer. More expressive uses:
- **Noise burst at attack** — route noise through a dedicated fast-decay envelope (or use Filter Env Amount at high positive value with short decay). This gives the percussive "click" at note onset that most organic sounds have.
- **Noise through resonant filter** — high resonance + noise = coloured noise peaks that read as formant-like. The bird call's "squaky" quality is likely this: broadband noise through a narrow resonant filter at the right frequency.
- **Velocity-sensitive noise injection** — louder notes get more noise, quieter notes are cleaner. Currently all notes render at the same noise level.

### C. Filter as primary shaper, not secondary
The filter in the current output is largely static. On the crane scream and most complex sounds:
- **Filter cutoff should sweep across the entire note.** Not just by a few percent — by 40–70% of its range. The full sweep from nearly-closed to fully-open over the note duration is what creates the "bloom" character of organic sounds.
- **Filter resonance adds formant peaks.** At Q ≈ 0.7–0.9 the resonance creates a single dominant frequency that tracks the filter sweep — this is what makes synthetic emulations of vocal sounds work.
- **Filter envelope must be decoupled from amp envelope.** The filter should open faster than the amp or close faster (or do both), creating a brightness arc that is completely independent of the volume arc. The current CMA-ES sometimes finds this but not reliably because the scores don't strongly reward it.

### D. Unison as a richness multiplier
OB-Xf's unison voices are the single cheapest quality improvement available. Four-voice unison with moderate detune (0.2–0.4 on Unison Detune) immediately makes any patch sound denser and more "analogue." The current pipeline found Unison Detune = 0.71 in one run — this is on the right track but should be a near-default rather than something discovered by CMA-ES.

---

## II. Better Target Analysis — Understanding What to Mimic

### A. Spectral envelope (pyworld SP) as a direct synthesis target
pyworld's `wav2world()` already returns SP (spectral envelope) alongside the F0 and AP we're already using. SP is the smooth spectral shape — essentially what a formant analyzer gives. This is directly the filter transfer function of the source and tells you:
- Where the spectral centroid is (maps to Filter Cutoff)
- How peaked the spectrum is (maps to Filter Resonance)
- How the shape changes over time (maps to Filter Envelope Amount + shape)

Using the per-frame SP distance as a scoring term would reward the optimizer for matching the filter character exactly, not just the average timbral colour.

**Implementation note.** S06b currently calls `pw.dio` + `pw.stonemask` (F0 only). The SP we want is one extra call on the existing F0:
```python
sp = pw.cheaptrick(audio.astype(np.float64), f0, t, sr)  # [n_frames, fft//2+1]
ap = pw.d4c(audio.astype(np.float64), f0, t, sr)         # already needed for §II.C
```
No new dependency, no model training. See [pyworld docs](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder).

### B. Attack transient decomposition
The burst events at 85ms/135ms are attack transients — very short energy spikes with their own spectral character. Currently they are detected as note regions (MIDI 84 for ~100ms) but the synthesizer's attack time governs whether they sound crisp or smeared. A dedicated transient analysis stage should:
1. Identify all transient events (< 50ms) by onset detection
2. Measure their peak frequency, duration, and spectral bandwidth
3. Use these to constrain Amp Env Attack (must be < transient_duration × 0.5) and Filter Env Attack

**Tooling.** [`librosa.onset.onset_detect`](https://librosa.org/doc/latest/generated/librosa.onset.onset_detect.html) (with `backtrack=True`) is the standard baseline. For sub-10ms accuracy on noisy / non-musical targets, [madmom's `CNNOnsetProcessor`](https://github.com/CPJKU/madmom) is the published benchmark winner — heavier dependency but trained on millions of onsets.

### C. Harmonic structure classification
The source sound's harmonic-to-noise ratio (which pyworld AP already gives) should drive oscillator selection more aggressively:
- AP mean < 0.2 across all bands → strongly harmonic → use saw or saw+pulse
- AP mean 0.2–0.5 → mixed → use saw+pulse + moderate noise
- AP mean > 0.5 → mostly noise → use noise-dominant patch with light oscillator

Currently this logic is implicit in the CMA-ES scoring. Making it an explicit gate (not just a soft reward) would eliminate entire classes of bad solutions.

### D. Inharmonicity fingerprinting
Ring modulation and cross-modulation create inharmonic sidebands at fc ± n×fm. The "squaky" quality of the crane scream is likely a narrow-band inharmonic component. Currently the optimizer may find this by accident (Ring Mod Volume got to 0.35 in one run) but it doesn't know WHY. An explicit inharmonicity measurement (deviation of spectral peaks from integer harmonic ratios) would give a direct objective for Cross Modulation and Ring Mod depth.

**Tooling.** [Parselmouth](https://github.com/YannickJadoul/Parselmouth) (Praat in Python) gives state-of-the-art formant tracking — locating 2–3 dominant resonance peaks over time. The deviation of these tracked peaks from integer multiples of F0 is the inharmonicity fingerprint we want.

---

## III. Better Scoring — What the Objective Function Needs to Reward

### A. Per-frame spectral envelope matching (SP distance)
Replace or supplement the time-averaged MRSTFT term with per-frame SP distance from pyworld. This directly measures whether the synthesizer's filter is tracking the source's filter movement at each point in time.

```
score_per_frame = cosine_dist(SP_source[t], SP_render[t])
score_sp = mean(score_per_frame)
```

Proposed composite: `0.40 × EnCodec + 0.25 × MRSTFT + 0.20 × AP + 0.15 × SP`

**Tooling.** [`auraloss`](https://github.com/csteinmetz1/auraloss) is already pinned in `requirements.txt` but never imported. `auraloss.freq.MultiResolutionSTFTLoss` is a battle-tested differentiable replacement for the hand-rolled MRSTFT in `s07_refine/audio_compare.py:53-100`. Drop-in swap, plus it unblocks DDSP-style differentiable analysis (§IV.A) since it returns gradients. SP itself comes from `pw.cheaptrick` — see §II.A.

### B. Pitch-invariant timbral scoring
EnCodec embeddings carry pitch information. When comparing a source at 1109Hz to a render at 1060Hz (a small pitch error), the cosine distance penalises the pitch error instead of the timbre error. A pitch-normalised embedding — computed by pitch-shifting the render to match the source before embedding — would give a cleaner timbral distance.

### C. Attack transient score
Add a dedicated score term for the attack transient: compare source and render in the first 50ms of each note region using spectral flux or a short-window SpectralCentroid distance. This would directly reward sharp attack character independent of the sustained-note timbre.

### D. Loudness-normalised comparison
LUFS-normalise source and render before all comparisons. Currently a render that is 6dB quieter than the source but otherwise identical scores worse than it should. This biases the optimizer toward solutions that happen to match the target level rather than the target timbre.

**Tooling.** [`pyloudnorm`](https://github.com/csteinmetz1/pyloudnorm) implements ITU-R BS.1770 LUFS metering in ~10 lines. Normalise both target and render to −23 LUFS before any embedding / MRSTFT / AP / SP comparison. Cheapest item in this section.

---

## IV. Better Optimisation — Moving Beyond CMA-ES

### A. Differentiable digital signal processing (DDSP)
The most direct quality improvement available in the literature. Rather than using a black-box synth (OB-Xf) and optimising its parameters with CMA-ES, build a differentiable approximation of the signal path:
- Differentiable oscillator bank (additive synthesis)
- Differentiable filter (biquad or Moog-style)
- Differentiable amplitude and filter envelopes

Google's DDSP library implements this. A DDSP reconstruction of the crane scream would give the "ideal" synthesis parameter trajectory — the continuous filter cutoff, gain envelope, and harmonic amplitudes over time — which can then be mapped to OB-Xf's discrete parameters.

DDSP as a "ground truth" analysis tool: run DDSP analysis on the target, extract its harmonic amplitudes and filter parameters, and use those as the initial guess (x0) for the CMA-ES rather than the current target analysis + surrogate path.

**Implementations.**
- [magenta/ddsp](https://github.com/magenta/ddsp) — the original (Engel et al., ICLR 2020, [paper](https://arxiv.org/abs/2001.04643)). TensorFlow.
- [acids-ircam/ddsp_pytorch](https://github.com/acids-ircam/ddsp_pytorch) — PyTorch port from IRCAM. Lighter dependency footprint, fits the existing torch stack.

**Most relevant recent work.** [DiffMoog](https://arxiv.org/abs/2401.12570) (Uzrad et al., ICASSP 2024) — a differentiable *modular* synthesizer where the signal-flow graph itself is differentiable. Architecturally the closest published analog to OB-Xa-class subtractive synths and the cleanest reference for §IV.B's "differentiable surrogate of OB-Xf" idea.

### B. Surrogate trained on richer data with temporal context
The current surrogate maps (params_15, note) → EnCodec_latent. Improvements:
- **Train on M=14 production data** (currently blocked on the capture run completing)
- **Include extra params** (Filter Env ADSR, Osc 2 Volume etc.) in the surrogate input so gradient inversion can reach them
- **Train on parameter trajectories, not static patches** — a surrogate that predicts the per-frame embedding from a parameter trajectory (not just a static vector) could drive the inversion of complex, time-varying sounds
- **Multi-task loss** — train on EnCodec + SP + AP simultaneously so the surrogate's gradients reflect all the qualities we care about

### C. Learned synthesis parameter prediction
Train a neural model directly: audio_embedding → synthesis_params. This replaces the gradient descent + CMA-ES loop with a single forward pass. Trade-off: requires a large dataset of (audio, params) pairs, which the capture stage is building. With M=14 × 16 notes × 16,384 vectors ≈ 4M samples, a small MLP or transformer could learn this mapping directly.

**Reference architecture.** [Sound2Synth](https://github.com/SymbioticLab/Sound2Synth) (Han et al., IJCAI 2022, [paper](https://arxiv.org/abs/2205.03390)) — CNN + Conformer encoder with per-parameter MLP heads, trained on DEXED. Directly portable to OB-Xf with the M=14 dataset. Output of this model becomes the warm-start `x0` for CMA-ES (§IV.D), reducing CMA-ES to a few-iteration polish rather than a full search.

For benchmarking against published baselines, [SpiegeLib](https://github.com/spiegelib/spiegelib) is the standard toolbox — implements hill-climb, GA, MFCC-MLP, LSTM, and multi-input-CNN baselines for synth parameter estimation.

### D. Hierarchical optimisation
Separate coarse from fine optimisation:
1. **Coarse** (discrete): Oscillator type, note interval for Osc 2, presence of ring mod, unison on/off. Try all combinations (16 at most). This replaces the current 3-config osc scouting with a richer discrete outer loop.
2. **Medium** (envelope shapes): Determine amplitude and filter envelope ADSR from the target's amplitude and spectral envelopes directly (pyworld + onset detection). Set these analytically, not by search.
3. **Fine** (CMA-ES): Search only the remaining parameters (filter cutoff region, noise level, cross-mod depth, detuning) with a much narrower prior, seeded by the coarse+medium results.

This shrinks the effective CMA-ES search space from 26 to perhaps 8–10 parameters, which converges 3–5× faster and more reliably.

---

## V. Synth Programming Knowledge — What the System Doesn't Know

### A. Synthesis conventions the optimizer should respect
The current system treats all parameters as continuous knobs with equal importance. A human synth programmer follows conventions that should be encoded as soft constraints or initialisation biases:

| Sound quality | Synthesis approach | Params to prioritise |
|---|---|---|
| Bright, nasal | High cutoff, low resonance, saw wave | Filter Cutoff > 0.6, Resonance < 0.4 |
| Dark, thick | Low cutoff, slow filter attack | Filter Cutoff < 0.3, Filter Env Amount < 0.2 |
| "Squaky"/inharmonic | Ring mod or cross-mod, high resonance | Ring Mod > 0.2, Resonance > 0.6 |
| Breathy/airy | High noise, open filter | Noise Volume > 0.2, Filter Cutoff > 0.5 |
| Percussive click | Fast attack, fast decay | Amp Env Attack < 0.05, Amp Env Decay < 0.2 |
| Pad/sustain | Slow attack, slow release | Amp Env Attack > 0.4, Amp Env Release > 0.4 |
| Vibrato | LFO to Osc 1 Pitch > 0 | LFO Rate + LFO Pitch Depth |
| Filter wah | LFO to Filter Cutoff | LFO Rate matches tempo/phrase |

Encoding these as a look-up driven by the target analysis (Q3: what is the dominant character?) would give dramatically better initial parameter guesses.

### B. Envelope-to-envelope correspondence
The amp envelope and filter envelope should not be optimised independently. The filter envelope should open the filter BEFORE or AS the amp envelope reaches peak, and close the filter slightly BEFORE the amp release ends (to prevent the resonant ping artifact already identified). These constraints are not in the current scoring and the CMA-ES finds them only by accident.

Explicit constraint: `Filter_Env_Release >= Amp_Env_Release` (already implemented as a hard bound) but also: `Filter_Env_Attack <= Amp_Env_Attack × 1.5` — the filter shouldn't open slower than the amp.

### C. Spectral centroid tracking via Filter Cutoff automation
The most important missing capability: per-frame Filter Cutoff automation following the source's spectral centroid trajectory. Currently Filter Cutoff is a static parameter (the same value for every frame). But the crane scream's spectral centroid drops from ~4kHz at the attack to ~800Hz during the sustain. A synthesizer replicating this needs the filter to sweep from open to closed over the note duration.

Implementation: [`librosa.feature.spectral_centroid`](https://librosa.org/doc/latest/generated/librosa.feature.spectral_centroid.html) returns a per-frame Hz array in one call. Map it through the offline calibration curve (§VI.B) to get Filter Cutoff in [0,1], and write that as a column in `stream_params.parquet` alongside the pitch bend. This is the single change most likely to make the output sound less "static." Pair with §VII priority-1.

---

## VI. Architecture Gaps to Fill

### A. Stage S09: Timbral decomposition
A new stage between target analysis (current S07 target_analysis.py) and CMA-ES initialisation:
1. **Harmonic analysis**: extract per-frame fundamental + harmonic amplitudes (via HPS or CREPE + pyworld SP)
2. **Formant tracking**: identify 2–3 dominant formant frequencies and their trajectories
3. **Noise floor profile**: per-band noise level over time (from pyworld AP)
4. **Synthesis mapping**: translate each of (harmonics, formants, noise) into OB-Xf parameter trajectories using the synth's measured transfer functions (calibrated offline)

This would replace the current heuristic target analysis (5 design questions) with a physically grounded decomposition.

**Tooling.** Each component has a mature off-the-shelf implementation:
- Harmonic / fundamental: `pw.cheaptrick` (SP) + `pw.dio`/`stonemask` (F0) — both already loaded.
- Formants: [Parselmouth](https://github.com/YannickJadoul/Parselmouth) (Praat) — `praat.call(sound, "To Formant (burg)", ...)`.
- Harmonic-vs-percussive split: [`librosa.effects.hpss`](https://librosa.org/doc/latest/generated/librosa.effects.hpss.html) — separates the tonal layer from the noise/transient layer before downstream analysis.
- Noise band profile: `pw.d4c` returns aperiodicity per band per frame.

The DDSP harmonic-plus-noise model (§IV.A) is the published reference for how (3) and (4) compose.

### B. Offline synth calibration (one-time)
To map source audio features to OB-Xf parameters accurately:
- Sweep Filter Cutoff 0→1, measure output spectral centroid → cutoff-to-Hz calibration curve
- Sweep Filter Resonance at various cutoff values, measure peak width → resonance-to-Q curve
- Sweep Osc 2 Pitch 0→1, measure output pitch shift → confirm the ±24st range
- Sweep Ring Mod Volume 0→1 with Osc 2 at various intervals, measure sideband positions
- Sweep LFO Rate 0→1, measure LFO frequency → rate calibration curve

These calibration tables replace the current "guess and score" approach with a lookup: "source has spectral centroid at 800Hz → set Filter Cutoff to 0.38."

**Tooling.** Beyond simple 1-D sweeps, [SALib](https://github.com/SALib/SALib) implements Sobol / Morris / FAST sensitivity analysis — useful for catching parameter *interactions* (e.g. Filter Cutoff calibration changes meaningfully under high Resonance). Same Sobol machinery you already use in `s03_dataset/sampling.py`, repurposed for measuring the synth instead of training the surrogate.

### C. Multi-instance render
For complex, layered sounds: render two independent OB-Xf instances simultaneously and mix. Instance A handles the harmonic/tonal layer; Instance B handles noise and transient content. The amp envelopes can be set differently for each (fast attack on noise, slower on harmonic). This doubles the parameter space but also the expressive range.

---

## VII. Priority Order for Implementation

| Priority | Change | Expected improvement | Effort |
|---|---|---|---|
| 1 | Per-frame Filter Cutoff automation from spectral centroid | Removes "static" character; most audible single change | Medium |
| 2 | Offline synth calibration curves | Accurate parameter → frequency mapping | Low |
| 3 | Osc 2 interval snapping (discrete octave/fifth/unison search) | Richer harmonic structure | Low |
| 4 | Spectral envelope (pyworld SP) as scoring term | Better filter matching | Medium |
| 5 | Attack transient constraint (Amp Env Attack < transient duration) | Correct attack character | Low |
| 6 | DDSP analysis to initialise CMA-ES x0 | Better convergence, fewer renders | High |
| 7 | Surrogate retrained on M=14 data with extra params | Better gradient inversion | High (blocked on S02) |
| 8 | Temporal surrogate (per-frame trajectories) | True dynamic synthesis | Very high |
| 9 | Multi-instance render | Richer layering | Medium |
| 10 | Learned synthesis parameter prediction (neural) | Fast inference, no CMA-ES | Very high (data-hungry) |

The first three items require no new ML training and could be implemented in days. They address the "too sparse" character directly. Items 4–5 refine the objective function. Items 6–10 are longer-term architectural changes.

**Library quick-wins (hours, not days).** Three foundational changes are pure library swaps and should land before anything else, because every later item benefits from them:

| Change | Library | Section |
|---|---|---|
| LUFS-normalise target and render before all scoring | [`pyloudnorm`](https://github.com/csteinmetz1/pyloudnorm) | §III.D |
| Replace hand-rolled MRSTFT with battle-tested differentiable version | [`auraloss`](https://github.com/csteinmetz1/auraloss) (already in `requirements.txt`, unused) | §III.A |
| Add SP and AP via existing pyworld call (`cheaptrick` + `d4c`) | pyworld (already loaded) | §II.A, §II.C, §III.A |
