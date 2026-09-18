#!/usr/bin/env bash
# ==============================================================================
# MoSPI PAIMANA (SIH 26103) — Automated Azure VM Production Deployment Script
# Deploys Prototype 2 on Port 8080 alongside Prototype 1 (Port 80)
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}==================================================================${NC}"
echo -e "${BLUE}    MoSPI PAIMANA SIH26103 — Azure VM Deployment (Project 2)      ${NC}"
echo -e "${BLUE}==================================================================${NC}"

# 1. Require root / sudo
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}[ERROR] This script must be run with sudo: sudo ./deploy_azure.sh${NC}" 
   exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_USER="${SUDO_USER:-$USER}"

echo -e "\n${YELLOW}[Step 1/5] Setting up Python virtual environment...${NC}"
VENV_PATH="${PROJECT_DIR}/.venv"
if [[ ! -d "$VENV_PATH" ]]; then
    sudo -u "$TARGET_USER" python3 -m venv "$VENV_PATH"
fi

sudo -u "$TARGET_USER" "$VENV_PATH/bin/pip" install --no-cache-dir --upgrade pip
sudo -u "$TARGET_USER" "$VENV_PATH/bin/pip" install --no-cache-dir -r "${PROJECT_DIR}/requirements.txt"

# 2. Build Frontend (Next.js starter if present)
FRONTEND_DIR="${PROJECT_DIR}/hackathon-frontend-starter"
if [[ -d "$FRONTEND_DIR" ]]; then
    echo -e "\n${YELLOW}[Step 2/5] Building Next.js frontend on port 3001...${NC}"
    cd "$FRONTEND_DIR"
    sudo -u "$TARGET_USER" npm install
    sudo -u "$TARGET_USER" npm run build
    cd "$PROJECT_DIR"
fi

# 3. Create systemd service for FastAPI backend (Port 8001)
echo -e "\n${YELLOW}[Step 3/5] Creating systemd service for PAIMANA FastAPI backend...${NC}"
cat <<EOF > /etc/systemd/system/paimana-backend.service
[Unit]
Description=MoSPI PAIMANA FastAPI Service (Project 2)
After=network.target

[Service]
User=${TARGET_USER}
Group=${TARGET_USER}
WorkingDirectory=${PROJECT_DIR}
ExecStart=${VENV_PATH}/bin/python -m uvicorn app.main:app \\
    --app-dir ${PROJECT_DIR}/backend \\
    --host 127.0.0.1 \\
    --port 8001 \\
    --workers 2
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=PORT=8001

[Install]
WantedBy=multi-user.target
EOF

# 4. Create systemd service for Frontend (Port 3001)
if [[ -d "$FRONTEND_DIR" ]]; then
cat <<EOF > /etc/systemd/system/paimana-frontend.service
[Unit]
Description=MoSPI PAIMANA Next.js Frontend Service (Project 2)
After=network.target

[Service]
User=${TARGET_USER}
Group=${TARGET_USER}
WorkingDirectory=${FRONTEND_DIR}
ExecStart=$(which npm) start -- -p 3001
Restart=always
RestartSec=5
Environment=NODE_ENV=production
Environment=PORT=3001
Environment=NEXT_PUBLIC_API_URL=http://127.0.0.1:8001

[Install]
WantedBy=multi-user.target
EOF
fi

# 5. Configure Nginx Reverse Proxy on Port 8080
echo -e "\n${YELLOW}[Step 4/5] Adding Port 8080 reverse proxy in Nginx...${NC}"

cat <<'EOF' > /etc/nginx/sites-available/paimana
server {
    listen 8080 default_server;
    listen [::]:8080 default_server;
    server_name _;

    client_max_body_size 50M;

    # Frontend UI (Next.js on port 3001)
    location / {
        proxy_pass http://127.0.0.1:3001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # FastAPI APIs & Docs
    location /api/ {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /docs {
        proxy_pass http://127.0.0.1:8001/docs;
        proxy_set_header Host $host;
    }

    location /openapi.json {
        proxy_pass http://127.0.0.1:8001/openapi.json;
        proxy_set_header Host $host;
    }

    # Built-in Executive Dashboard
    location /dashboard {
        proxy_pass http://127.0.0.1:8001/dashboard;
        proxy_set_header Host $host;
    }

    location /static/ {
        proxy_pass http://127.0.0.1:8001/static/;
        proxy_set_header Host $host;
    }
}
EOF

ln -sf /etc/nginx/sites-available/paimana /etc/nginx/sites-enabled/paimana
nginx -t
systemctl restart nginx

# Reload and start systemd services
systemctl daemon-reload
systemctl enable --now paimana-backend
if [[ -d "$FRONTEND_DIR" ]]; then
    systemctl enable --now paimana-frontend
fi

echo -e "\n${GREEN}==================================================================${NC}"
echo -e "${GREEN}🎉 PROJECT 2 DEPLOYED SUCCESSFULLY! Both projects running!       ${NC}"
echo -e "${GREEN}==================================================================${NC}"
PUBLIC_IP=$(curl -s ifconfig.me || hostname -I | awk '{print $1}')
echo -e "\n  👉 Project 1 (MarSlick):      ${BLUE}http://${PUBLIC_IP}/${NC}"
echo -e "  👉 Project 2 (MoSPI PAIMANA): ${BLUE}http://${PUBLIC_IP}:8080/${NC}"
echo -e "  👉 Project 2 Swagger Docs:    ${BLUE}http://${PUBLIC_IP}:8080/docs${NC}"
echo -e "\nEnsure Port 8080 is allowed in your Azure Network Security Group!"
echo -e "${GREEN}==================================================================${NC}\n"
