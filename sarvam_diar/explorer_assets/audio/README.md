# audio/

Empty by default, and deliberately so: the corpus is ~1.3 GB of WAV, which does
not belong in a git repository.

The UI looks for `audio/<clip_id>.wav`. Without it everything still works except
playback -- the timeline, error regions, filtering and speaker rows are all
driven by the JSON in `../data/`.

To add audio, either:

    # copy the whole corpus (~1.3 GB)
    cp /path/to/sarvam_diarization/audio_16k/*.wav audio/

    # or just the clips you are inspecting
    cp /path/to/audio_16k/Cku_X_SL7qU__60_660.wav audio/

or re-export with `explorer.export(..., copy_audio=True)`.
