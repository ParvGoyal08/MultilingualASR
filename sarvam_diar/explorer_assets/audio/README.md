# audio/

Empty by default, and deliberately so: the corpus is ~1.3 GB of WAV, which does
not belong in a git repository.

The UI looks for `audio/<clip_id>.wav`. Without it everything still works except
playback -- the timeline, error regions, filtering and speaker rows are all
driven by the JSON in `../data/`.

To add audio, symlink rather than copy -- the export is a view of the corpus,
not a second copy of it, and `python -m http.server` follows symlinks fine:

    ln -sf /path/to/sarvam_diarization/audio_16k/*.wav audio/

Copying also works if the export has to be moved to another machine:

    cp /path/to/sarvam_diarization/audio_16k/*.wav audio/     # ~1.3 GB
    cp /path/to/audio_16k/Cku_X_SL7qU__60_660.wav audio/      # or just one

or re-export with `explorer.export(..., copy_audio=True)`.

Note that `has_audio` in `data/clips.json` is recorded when the export is
WRITTEN. If audio is added afterwards the flag is stale; it only affects
reporting, not playback.
