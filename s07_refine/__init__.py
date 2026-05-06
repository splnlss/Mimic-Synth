"""s07_refine — close the surrogate-to-real-synth gap via real VST renders.

Modules:
  audio_compare    shared rendering + scoring helpers
  vst_hill_climb   Strategy 1: per-param coordinate-descent on real renders
  vst_cmaes        Strategy 2: CMA-ES on real renders (planned)

See build_instructions/07 Refine VST Loop.md for design rationale.
"""
