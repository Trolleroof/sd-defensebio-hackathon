FROM python:3.13.3-slim-bookworm@sha256:56a11364ffe0fee3bd60af6d6d5209eba8a99c2c16dc4c7c5861dc06261503cc
COPY requirements.lock /opt/research/requirements.lock
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install --timeout 120 --retries 5 pip==26.2.1 && pip install --timeout 120 --retries 5 --resume-retries 10 -r /opt/research/requirements.lock
ENV CUDA_VISIBLE_DEVICES="" MPLCONFIGDIR=/tmp/matplotlib PYTHONDONTWRITEBYTECODE=1
WORKDIR /work
CMD ["python"]
