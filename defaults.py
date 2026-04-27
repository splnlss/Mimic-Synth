"""
Project-wide defaults for MimicSynth.

Production code should prefer reading from the profile YAML (e.g.
profile["probe"]["sample_rate"]) when a profile is available. These
constants exist for scripts that run before a profile is loaded
(enumerate_params.py) and for unit tests that don't use a real profile.
"""

SAMPLE_RATE = 48000
BUFFER_SIZE = 512
