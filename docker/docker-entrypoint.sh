#!/bin/bash
set -e

export DISPLAY=:100
geometry=${FLIMKIT_GEOMETRY:-1440x900}
export FLIMKIT_WEB_PASSWORD=${FLIMKIT_WEB_PASSWORD:-$FLIMKIT_PASSWORD}

# Headless X server: the Tk window runs here and the web UI drives it
Xvfb :100 -screen 0 "${geometry}x24" -nolisten tcp >/tmp/xvfb.log 2>&1 &

for i in $(seq 1 100); do
    xdpyinfo -display :100 >/dev/null 2>&1 && break
    sleep 0.2
done

if [ "${FLIMKIT_DESKTOP:-0}" = "1" ]; then
    # Someone can now see the desktop, so its dialogs only move to the web page while the page is in use
    export FLIMKIT_WEB_HEADLESS=0
    fluxbox >/tmp/fluxbox.log 2>&1 &
    if [ -n "$FLIMKIT_PASSWORD" ]; then
        x11vnc -storepasswd "$FLIMKIT_PASSWORD" /tmp/vncpass >/dev/null 2>&1
        auth_args=(-rfbauth /tmp/vncpass)
    else
        auth_args=(-nopw)
    fi
    x11vnc -display :100 -forever -shared -localhost -rfbport 5900 "${auth_args[@]}" -bg -o /tmp/x11vnc.log
    websockify --web=/usr/share/novnc "${FLIMKIT_DESKTOP_PORT:-14501}" localhost:5900 >/tmp/websockify.log 2>&1 &
fi

if [ -z "$FLIMKIT_WEB_PASSWORD" ]; then
    echo "[flimkit] FLIMKIT_PASSWORD is not set: anyone who can reach port ${FLIMKIT_WEB_PORT} can use FLIMKit and browse the mounted data." >&2
fi

exec python /app/main.py
