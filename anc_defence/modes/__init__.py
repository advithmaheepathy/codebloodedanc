"""Operating modes.

``offline_file``          process pre-recorded WAVs, full metrics, PDF report
``live_dfn_only``         live single microphone, neural stage only (simplest path)
``live_injected_noise``   live single microphone with a noise file mixed in digitally;
                          the same file, RIR-free, is the NLMS reference

Modes differ only in where samples come from. All of them use
:class:`anc_defence.pipeline.Pipeline` for the processing itself.
"""
