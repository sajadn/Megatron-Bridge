#!/usr/bin/env bash
# Sync local Megatron-Bridge to core cluster (cw-dfw-cs-001-login-02)
# Uses the 'core' alias from ~/.ssh/config (IdentityFile: ~/.ssh/id_rsa)

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_HOST="core"
REMOTE_DIR="/lustre/fsw/portfolios/coreai/users/linnanw/Megatron-Bridge"

rsync -avz --progress \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.egg-info/' \
    --exclude='.venv/' \
    --exclude='dist/' \
    --exclude='build/' \
    --exclude='*.DS_Store' \
    --exclude='transformers' \
    --exclude='diffusers' \
    --exclude='JustGRPO' \
    -e "ssh" \
    "${LOCAL_DIR}/" \
    "${REMOTE_HOST}:${REMOTE_DIR}/"