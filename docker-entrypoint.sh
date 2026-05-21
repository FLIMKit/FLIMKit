#!/bin/bash
set -e

exec xpra start :100 \
    --start-child="python /app/main.py" \
    --exit-with-children=yes \
    --html=on \
    --bind-tcp=0.0.0.0:14500 \
    --no-daemon \
    --pulseaudio=no \
    --notifications=no \
    --bell=no \
    --mdns=no \
    --open-files=no \
    --printing=no
