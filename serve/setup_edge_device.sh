#!/bin/bash
# Edge device one-time setup script.
# Configures Docker to trust the training workstation's registry (insecure, LAN-only)
# and installs an optional systemd service for auto-start on boot.
#
# Usage:
#   bash setup_edge_device.sh [REGISTRY_HOST] [REGISTRY_PORT] [IMAGE_TAG]
#
# Example:
#   bash setup_edge_device.sh workstation 5000 workstation:5000/yolo-inference-abc12345:20260406

set -e

REGISTRY_HOST="${1:-workstation}"
REGISTRY_PORT="${2:-5000}"
IMAGE_TAG="${3:-}"

echo "=== Edge device setup ==="
echo "Registry: ${REGISTRY_HOST}:${REGISTRY_PORT}"

# 1. Configure insecure registry
DAEMON_JSON=/etc/docker/daemon.json

if [ -f "$DAEMON_JSON" ]; then
    # Merge rather than overwrite if file already exists
    python3 -c "
import json, sys
with open('$DAEMON_JSON') as f:
    cfg = json.load(f)
reg = '${REGISTRY_HOST}:${REGISTRY_PORT}'
regs = cfg.get('insecure-registries', [])
if reg not in regs:
    regs.append(reg)
cfg['insecure-registries'] = regs
print(json.dumps(cfg, indent=2))
" > /tmp/daemon.json.tmp && mv /tmp/daemon.json.tmp "$DAEMON_JSON"
else
    mkdir -p /etc/docker
    cat > "$DAEMON_JSON" <<EOF
{
  "insecure-registries": ["${REGISTRY_HOST}:${REGISTRY_PORT}"]
}
EOF
fi

echo "Restarting Docker..."
systemctl restart docker

echo "Docker configured to trust ${REGISTRY_HOST}:${REGISTRY_PORT}"

# 2. Optional: install systemd service for auto-start
if [ -n "$IMAGE_TAG" ]; then
    SERVICE_FILE=/etc/systemd/system/yolo-inference.service

    cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=YOLO Inference ROS2 Node
After=docker.service network-online.target
Requires=docker.service

[Service]
Type=simple
Restart=on-failure
RestartSec=10
ExecStartPre=-/usr/bin/docker stop yolo-inference
ExecStartPre=-/usr/bin/docker rm yolo-inference
ExecStart=/usr/bin/docker run --rm --name yolo-inference \\
  --gpus all --network host \\
  -v yolo-cache:/model/cache \\
  ${IMAGE_TAG}
ExecStop=/usr/bin/docker stop yolo-inference

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable yolo-inference
    echo "Systemd service installed: yolo-inference.service"
    echo "Start with: systemctl start yolo-inference"
    echo "Logs with:  journalctl -u yolo-inference -f"
fi

echo "=== Setup complete ==="
