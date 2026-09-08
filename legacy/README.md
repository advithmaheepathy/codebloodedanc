# Legacy reference material

Kept for reference only. Nothing here is on the runtime path.

## `nlms.pdf`

The original MATLAB NLMS experiment: primary = speech + gunshot, reference = the
gunshot recording itself, 16 kHz, `filterLength = 1`, `mu = 0.05`.

Two things about it do not carry over to this project, and the current design
deliberately avoids both:

1. **The reference was the identical, sample-aligned noise recording.** Cancelling
   it needs no acoustic path modelling at all, so ERLE measured that way is a
   tautology. In this project the noise is convolved with a room impulse response
   before it reaches the primary channel, while the NLMS reference stays dry, so the
   filter has to identify a real path.
2. **A single tap is a scalar gain**, not a filter. The current default is 3840 taps
   (80 ms at 48 kHz), chosen to comfortably exceed the significant length of the
   simulated RIRs.

## `colab_deepfilternet_smoke.py`

The original Colab notebook export (`Untitled1.ipynb`): builds a Python 3.11 venv,
installs `deepfilternet==0.5.6`, and runs the pretrained model's offline
`enhance()` on one uploaded file. It confirmed the pretrained model works before
this repository existed. The Python 3.11 pin is not incidental: `deepfilternet`
0.5.6 requires `numpy>=1.22,<2.0`, which rules out newer interpreters. The same
constraint is why this project targets 3.11.
