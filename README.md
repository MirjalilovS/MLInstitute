# MNIST Digit Recogniser — End-to-End Demo

Draw a digit in your browser, press **Submit**, and a Convolutional Neural Network predicts the digit and logs the result in PostgreSQL.

**Live demo → <http://195.201.231.19:8501/>**  
(running on a Hetzner CX11 VPS)

---

## Features

| Layer | Tech | Highlights |
|-------|------|------------|
| **UI** | Streamlit + streamlit-drawable-canvas | Free-hand canvas, instant confidence percentage, responsive layout |
| **Model** | PyTorch CNN (SimpleCNN) | 99 % test accuracy on MNIST, CPU-only, small weight file |
| **API / Logging** | psycopg2 | Stores `predicted`, `confidence_pct`, `true_digit`, `ts` |
| **Database** | PostgreSQL 15 (Docker) | Auto-migrating schema, persistent volume |
| **Infrastructure** | Docker + Compose v2 | One-command start, reproducible anywhere |
| **Deployment** | Hetzner Cloud CX11 | Ubuntu 22.04, firewall open only on 22 and 8501 |

---

## Quick Start (Docker)

```bash
git clone https://github.com/<your-fork>/mnist-end2end.git
cd mnist-end2end

# 1.  (Optional) adjust credentials
cp .env.example .env              # defaults: mnist_user / mnist_pw

# 2.  Build & start
docker compose up -d --build

# 3.  Open the app
open http://localhost:8501
