#!/bin/bash
set -e

export DISPLAY=:100
geometry=${FLIMKIT_GEOMETRY:-1440x900}

# Headless X server
Xvfb :100 -screen 0 "${geometry}x24" -nolisten tcp >/tmp/xvfb.log 2>&1 &

# Wait for the X server to accept connections
for i in $(seq 1 100); do
    xdpyinfo -display :100 >/dev/null 2>&1 && break
    sleep 0.2
done

fluxbox >/tmp/fluxbox.log 2>&1 &

if [ -n "$FLIMKIT_PASSWORD" ]; then
    x11vnc -storepasswd "$FLIMKIT_PASSWORD" /tmp/vncpass >/dev/null 2>&1
    auth_args=(-rfbauth /tmp/vncpass)
else
    auth_args=(-nopw)
fi


x11vnc -display :100 -forever -shared -rfbport 5900 "${auth_args[@]}" -bg -o /tmp/x11vnc.log

websockify --web=/usr/share/novnc 14500 localhost:5900 >/tmp/websockify.log 2>&1 &

exec python /app/main.py
