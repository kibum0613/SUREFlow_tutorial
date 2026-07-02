#!/usr/bin/env bash
set -e

echo "=== Python processes ==="
ps aux | grep python | grep -v grep || true

echo
echo "=== RAM ==="
free -h

echo
echo "=== GPU ==="
nvidia-smi || true

echo
echo "=== Recent checkpoints ==="
find ~/projects/SUREFlow/logs -type f -name "*.pth" 2>/dev/null | tail || true

echo
echo "=== Recent evaluation videos ==="
find ~/projects/SUREFlow/logs -type f -name "*.mp4" 2>/dev/null | tail || true
