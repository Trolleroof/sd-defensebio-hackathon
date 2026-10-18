"""Participant image spec: same digest-pinned Python 3.13.3 base and lock as
research_env/Dockerfile. Running this module BUILDS on Beam (billable); it is a
one-time user step documented in docs/BEAM_SETUP.md.

    python -m research_env.beam.image          # prints the Beam image id
"""
from __future__ import annotations
import sys

from research_env.beam.config import PIP_VERSION, BeamConfig


def participant_image(cfg: BeamConfig, sdk):
    # beam-client's PythonVersion enum stops at 3.12; leaving python_version unset
    # makes Beam use the base image's own interpreter (3.13.3 here).
    return (sdk.Image.from_registry(cfg.base_image)
            .add_commands([f'python -m pip install --timeout 120 --retries 5 pip=={PIP_VERSION}'])
            .add_python_packages(str(cfg.requirements_lock))
            .with_envs({'CUDA_VISIBLE_DEVICES': '-1', 'MPLCONFIGDIR': '/tmp/matplotlib',
                        'PYTHONDONTWRITEBYTECODE': '1'}))


def main(argv=None):  # pragma: no cover - performs a billable remote build
    import beam
    result = participant_image(BeamConfig.from_env(), beam).build()
    if not result.success:
        print(f'image build failed: {getattr(result, "error", "")}', file=sys.stderr)
        return 1
    print(result.image_id)
    return 0


if __name__ == '__main__':
    sys.exit(main())
