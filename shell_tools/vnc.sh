#!/bin/bash

# Virtual X framebuffer for running GUI without a display
Xvfb :20 -screen 0 1366x768x16 &

# Start the VNC server with password Silcam
x11vnc -passwd Silcam -display :20 -N -forever &

# Launch xterm, available via VNC
xterm &
