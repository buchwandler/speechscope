"""WAV input boundary. No implicit trimming or time changes."""

from pathlib import Path

import numpy as np

from .errors import InvalidAudioError
from .types import AudioInput


def load_wav(path: Path | str) -> AudioInput:
    source = Path(path)
    if source.suffix.lower() != ".wav" or not source.is_file():
        raise InvalidAudioError(f"Expected a readable WAV file: {source}")
    try:
        import soundfile as sf

        info = sf.info(source)
        if (
            info.format != "WAV"
            or info.channels not in (1, 2)
            or info.samplerate <= 0
            or info.frames <= 0
        ):
            raise InvalidAudioError(
                "Audio must be nonempty mono/stereo WAV with positive sample rate"
            )
        samples, rate = sf.read(source, dtype="float64", always_2d=True)
        if samples.size == 0 or not np.isfinite(samples).all():
            raise InvalidAudioError("WAV contains no valid finite samples")
    except InvalidAudioError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise InvalidAudioError(f"Cannot read WAV {source}: {exc}") from exc
    channels = samples.shape[1]
    from audiosig import downmix_to_mono

    mono = (downmix_to_mono(samples, channel_axis=1) if channels == 2 else samples[:, 0]).astype(
        np.float64
    )
    # Audiosig downmix averages channels; it does not clip or trim the signal.
    return AudioInput(
        samples=mono,
        sample_rate=int(rate),
        channels=channels,
        duration_s=len(mono) / rate,
        path=source,
        transforms=("average_channels",) if channels == 2 else (),
    )
