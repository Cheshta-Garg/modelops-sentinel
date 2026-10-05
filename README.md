# ModelOps Sentinel

> **Production ML Model Lifecycle, Serving & Monitoring Platform**

ModelOps Sentinel is an engineering-first, production-oriented Machine Learning platform designed to manage, serve, observe, and protect ML models in high-throughput environments.

---

## Architectural Principles

1. **Modular Asynchronous Monolith**: Single codebase, two processes (`API` and `Worker`), single Docker image. Zero unnecessary microservice overhead.
2. **Durable vs. Ephemeral Separation**:
   - **PostgreSQL**: Durable source of truth for models, versions, deployments, audit records.
   - **Redis**: High-speed inference caching, task queue broker, rate limiting, and telemetry buffering.
3. **Observability & Guardrails**: Built-in liveness (`/health`) and readiness (`/ready`) probes, structured JSON logging with correlation IDs, Prometheus telemetry, and out-of-band statistical drift detection.
4. **Release Engineering**: Canary traffic splitting with automated rollback mechanisms on error/latency anomalies.

---

## Project Structure

```text
modelops-sentinel/
├── app/
│   ├── api/             # HTTP endpoints and routing (v1)
│   ├── core/            # Config, logging, error handling, middleware
│   └── main.py          # FastAPI application factory & lifecycle
├── docs/                # Comprehensive stage-by-stage documentation
├── tests/               # Pytest test suite (unit & integration)
├── .env.example         # Environment template
├── docker-compose.yml   # Multi-container orchestration (Postgres, Redis, API)
├── Dockerfile           # Multi-stage production build
└── pyproject.toml       # Linter (Ruff) and test configurations
```

---

## Quickstart (Local Development)

### 1. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### 2. Run Quality Checks
```bash
ruff check .
pytest
```

### 3. Start the API Locally
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
- Interactive API Docs: `http://localhost:8000/docs`
- Liveness Probe: `http://localhost:8000/api/v1/health`
- Readiness Probe: `http://localhost:8000/api/v1/ready`

### 4. Docker Deployment
```bash
docker compose up -d
```
