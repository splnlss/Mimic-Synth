---
tags: [build, 08-package, packaging, deployment, ux]
created: 2026-04-19
---

# 08 Package the Tool Overview

> [!info] Goal
> Turn the previous seven buckets' code and artefacts into a usable tool — a CLI, a local web UI, and/or a realtime plug-in. Make the happy-path flow "drop in an audio file or plug in a mic, get a patch / stream" reproducible by anyone, not just the person who wrote the pipeline.

Bucket 8 is a deliberate forcing function. Until you package the pipeline into something someone else can run, you don't actually know whether steps 1–7 are robust. Packaging exposes assumption leaks and missing calibration steps ruthlessly.

## Where Bucket 8 sits

```
Buckets 1–7 artefacts           Bucket 8 (this doc)           Users
  profiles/                        ┌────────────────────┐
  surrogate/<synth>/    ─────►    │  CLI              │    ─────► scripted / batch use
  inverse/<synth>/      ─────►    │  Web UI (Gradio)  │    ─────► interactive offline
  patches/              ─────►    │  Max / Pd patch   │    ─────► realtime performance
  cache/                ─────►    │  Neutone plug-in  │    ─────► DAW integration
                                   └────────────────────┘
```

Four deliverable surfaces, one shared core library. Not all four are required for v1 — pick the subset that matches your use case.

## Prerequisites

- [[05 Surrogate|Bucket 5]] + [[06 Invert|Bucket 6]] working for at least one synth (offline path).
- [[06b Live|Bucket 6b]] if you want the realtime surfaces (Max / Neutone).
- [[07 Refine|Bucket 7]] optional but recommended for "production-quality" offline output.

## Minimum viable UX — offline

The happy path:

1. User drops an audio file into a file picker.
2. User selects a synth profile from a dropdown.
3. Tool runs embedding → inverse → surrogate-search → (optional) hardware refinement.
4. Tool returns a patch (loadable onto the synth), an A/B audio comparison, and a similarity score.

That's it. No manual hyperparameter tuning, no config files, no training. Internally it's every previous bucket strung together; externally it's five clicks.

## Offline surface: CLI + Gradio

A shared core library, two thin front-ends.

### Shared core

```python
# synthmimic/api.py
from .profile      import load_profile
from s04_embed     import Embedder
from s05_surrogate import Surrogate
from s06_invert    import invert
from s02_capture   import CaptureLoop

def mimic(target_wav: Path, synth_id: str,
          refine: bool = False) -> dict:
    """One-shot offline mimic pipeline."""
    profile   = load_profile(f"s01_profiles/{synth_id}.yaml")
    surrogate = load_surrogate(f"surrogate/{synth_id}/latest.pt", profile)
    inverse   = load_inverse(f"inverse/{synth_id}/latest.pt", profile)
    embedder  = Embedder()

    patch, score = invert(target_wav, profile, surrogate, embedder,
                          inverse_seed=inverse)

    if refine:
        with CaptureLoop(profile) as cap:
            patch, score = hw_refine(cap, embedder, patch, score, profile)

    rendered_wav = render_with_patch(patch, profile)

    return {
        "patch_yaml":   patch.to_yaml(),
        "patch_syx":    patch.to_sysex(),
        "target_wav":   target_wav,
        "rendered_wav": rendered_wav,
        "score":        score,
    }
```

### CLI

```bash
synthmimic mimic bird.wav --synth obxf
synthmimic mimic bird.wav --synth novation_peak --refine
synthmimic list-synths
synthmimic profile-doctor --synth novation_peak    # verify profile matches the plugged-in synth
```

Use [click](https://click.palletsprojects.com/) or [typer](https://typer.tiangolo.com/). Keep the CLI as thin as possible — all the heavy lifting is in `synthmimic.api`.

### Web UI with Gradio

```python
# synthmimic/ui.py
import gradio as gr
from .api import mimic

def run(wav_file, synth_id, refine):
    result = mimic(Path(wav_file.name), synth_id, refine=refine)
    return (result["target_wav"], result["rendered_wav"],
            result["patch_yaml"], f"{result['score']:.4f}")

iface = gr.Interface(
    fn=run,
    inputs=[gr.File(label="Target sound (WAV)"),
            gr.Dropdown(list_synths(), label="Synth profile"),
            gr.Checkbox(label="Refine on real hardware")],
    outputs=[gr.Audio(label="Target"), gr.Audio(label="Synth mimic"),
             gr.Textbox(label="Patch YAML"), gr.Textbox(label="Cosine distance")],
    title="Synth Mimic",
)

if __name__ == "__main__":
    iface.launch()
```

Gradio at a local URL is enough. [Streamlit](https://streamlit.io/) is equivalent if the team's background leans that way. [FastAPI](https://fastapi.tiangolo.com/) + a custom front-end is the right answer only if you're building for third-party users.

## Realtime surface: Max / Pd via nn~

The [[06b Live|Bucket 6b]] streaming path deploys directly as an [nn~](https://github.com/acids-ircam/nn_tilde) Max object.

Export flow:

```python
# s08_package/export_nn_tilde.py
import torch
# Wrap the streaming inverse + pitch tracker + smoother into a single forward model.
model = LiveMimicModel(inverse, pitch_tracker, smoother).eval()
torch.jit.script(model).save("synthmimic_live.ts")
```

The `.ts` file drops into a Max patch as `[nn~ synthmimic_live.ts]`. Audio in, CC out. The Max patch handles:

- Audio input routing (adc~ → nn~).
- MIDI CC scheduling (the `ctlout` family of objects).
- Mode selection (Follow vs Echo) via simple `[gate]` logic.

The nn~ side is minimal — most of the engineering effort is upstream. Keep a reference Max patch in the repo as a template.

## DAW surface: Neutone plug-in

[Neutone SDK](https://github.com/Neutone/neutone_sdk) wraps a PyTorch model as a VST3 / AU plug-in so the whole "audio in → params → MIDI" loop runs inside Ableton, Logic, Reaper, etc.

Two caveats up front:

- Neutone's model interface is opinionated — samples in, samples out. The "MIDI out" path is not its native case. As of writing, the workaround is to emit a special audio signal that the DAW routes to a MIDI plug-in (e.g. a control-voltage-to-MIDI converter). Document this carefully in user docs.
- Latency through a plug-in chain is the DAW's buffer size × 2 at minimum. Expect 10–30 ms extra on top of Bucket 6b's 30 ms budget. Fine for studio work, marginal for live.

If neither caveat kills the use case, Neutone makes distribution trivial: one plug-in file per platform, any DAW.

## Patch export formats

Every synth patch should be exportable in at least three forms so users can load it wherever they want:

- **`patch.yaml`** — human-readable, the source of truth. Contains normalised `[0,1]` values, profile ID, firmware version, timestamp.
- **`patch.syx`** — for synths with SysEx patch dump (Juno-106, Peak, many others). Ready to drop into [SysEx Librarian](https://www.snoize.com/SysExLibrarian/) / [MIDI-OX](http://www.midiox.com/).
- **`patch.json` CC dump** — for synths controlled via CC (Peak, many modern). A flat `{cc_number: value}` dict playable via any MIDI utility.

Bucket 1 profile metadata tells the exporter which subset of these is meaningful for a given synth.

## Distribution

For v1 keep it simple:

- Source install from a GitHub repo: `pip install -e .` plus a `setup-cowork.sh` / `setup.ps1` that installs OS-specific deps (ASIO driver pointers, JUCE 8 runtime, etc.).
- Pre-trained weights for each supported synth hosted as GitHub release assets (or Hugging Face if weights get large).
- A short Getting Started doc that walks through: install, download weights for your synth, run `synthmimic mimic --synth obxf bird.wav`.

Packaging as a standalone `.app` / `.exe` / `.deb` is post-v1 — premature until the core workflow is settled.

## Documentation

A single `README.md` + this `Build Instructions/` folder is enough for v1. Recommended structure:

```
docs/
├── getting-started.md        # 15-min quickstart
├── adding-a-synth.md         # how to write a Bucket 1 profile
├── architecture.md           # points to Synth-Mimic-Pipeline.md
└── troubleshooting.md        # "silent capture", "MIDI port not found", etc.
```

Don't write API docs manually — use [mkdocs-material](https://squidfunk.github.io/mkdocs-material/) with docstring extraction.

## Versioning and releases

- Semantic versioning on the Python package.
- **Dataset / weights versioned separately** from code. A surrogate trained on dataset `obxf_2026-04` doesn't work against dataset `obxf_2026-05`; be explicit in the filenames.
- Release notes call out: which synths are supported, which weights are shipped, what changed in the profile format.

## Validation

Before declaring Bucket 8 done:

1. **Fresh-machine install.** On a machine that has never seen the project, clone the repo, follow `getting-started.md`, run the CLI against the shipped bird example. Fix everything that's missing or confusing.
2. **Every surface works end-to-end.** CLI, Gradio, nn~ patch, and (if chosen) Neutone plug-in each take a target and produce a result.
3. **Happy-path latency.** Offline mimic of a 3-second target completes in under ~5 seconds (excluding optional hardware refinement). Live mode end-to-end latency in the 100–300 ms range (see [[06b Live]] for the full budget breakdown).
4. **Cross-platform smoke test.** At minimum: CLI + Gradio on macOS and Windows. The realtime surfaces can be platform-specific for v1.

## Dependencies

```bash
pip install click typer gradio fastapi uvicorn
# realtime deployment
pip install neutone_sdk torch
# docs
pip install mkdocs mkdocs-material mkdocstrings-python
```

## Uncertainties to flag

- **Neutone "audio-to-MIDI" is awkward.** Their SDK centres on audio in / audio out. Emitting MIDI from a Neutone plug-in requires a workaround (audio signal → MIDI converter plug-in). This may fundamentally limit Neutone's usefulness for this project. Verify early; fall back to nn~ + Max if blocking.
- **Packaging ASIO dependencies on Windows.** ASIO drivers are per-device and require user-provided driver installation — there's no pip package for "ASIO support." Document this as a prerequisite rather than trying to bundle.
- **Weights hosting will grow fast.** EnCodec checkpoint is ~80 MB. Per-synth surrogates are small (~5 MB) but accumulate; inverse models add another ~5 MB each. Plan for Hugging Face / S3 hosting once total package exceeds ~200 MB.
- **Max / Pd licensing.** Pure Data is free; Max is paid. Documenting both is fine but they have different audiences. For a general-audience tool, Pd is the more inclusive default.
- **"Profile doctor" is more work than it sounds.** A CLI that verifies a profile matches the plugged-in synth needs to send test MIDI and correlate audio response — basically a mini Bucket 2. Worth doing because profile/synth mismatches are a common failure mode, but don't underestimate it.
- **Live use needs per-user calibration.** Latency measurement from Bucket 2 V2 can't be shipped as a constant — every user's setup has different latency. Bake the calibrate_latency step into the first-run setup of the realtime surface.
- **Gradio vs Streamlit is a style call.** Both work. Gradio's file-drop + audio player is slightly more polished for audio work; Streamlit's layout primitives are more flexible. Pick one and don't look back in v1.

## When Bucket 8 is done

- A fresh-machine install works end-to-end from README alone.
- At least one surface (CLI preferred) runs the full offline mimic pipeline in under ~5 s per target.
- Patches export in the formats meaningful for each supported synth.
- If realtime: at least one of nn~ / Neutone runs with measured end-to-end latency below the [[06b Live|Bucket 6b]] budget.
- Weights + dataset manifests are versioned and reproducible.

## References

- [Gradio](https://gradio.app/) · [Streamlit](https://streamlit.io/) · [FastAPI](https://fastapi.tiangolo.com/)
- [Click](https://click.palletsprojects.com/) · [Typer](https://typer.tiangolo.com/)
- [nn~ / nn_tilde](https://github.com/acids-ircam/nn_tilde) · [Neutone SDK](https://github.com/Neutone/neutone_sdk)
- [mkdocs-material](https://squidfunk.github.io/mkdocs-material/)
- [SysEx Librarian (macOS)](https://www.snoize.com/SysExLibrarian/) · [MIDI-OX (Windows)](http://www.midiox.com/)
- [[Synth-Mimic-Pipeline]] · [[06 Invert]] · [[06b Live]] · [[07 Refine]]
