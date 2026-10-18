#!/bin/sh
# Run the linux-pod backend tests in a Linux container.
#   default: RunPod-like container (root, default caps, so no network namespace)
#   --netns: add CAP_SYS_ADMIN so the network-namespace path is exercised too
#   PLATFORM=linux/amd64: test the x86_64 seccomp program (RunPod hosts) on an ARM Mac
set -eu
caps="--platform ${PLATFORM:-linux/$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/')}"
[ "${1:-}" = "--netns" ] && caps="$caps --cap-add SYS_ADMIN --security-opt seccomp=unconfined"
# shellcheck disable=SC2086
docker run --rm $caps -v "$(pwd)":/src:ro -w /src python:3.13-slim \
    sh -c 'cp -r /src /repo && cd /repo && python -m unittest tests.test_linux_pod_backend -v'
