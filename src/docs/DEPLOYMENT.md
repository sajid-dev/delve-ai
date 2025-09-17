# Deployment Guide

This guide explains how to deploy the LLM chat backend in a production environment.

1. **Install Python dependencies** using the pinned versions in `pyproject.toml`.
2. **Set environment variables** by copying `.env.example` to `.env` and providing your OpenAI API key.
3. **Run the ASGI server** with an appropriate process manager such as `uvicorn` or `gunicorn`.  For example:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

4. **Configure logging** by ensuring the `logs/` directory is writable.  Log rotation and retention are handled automatically.
5. **Persist the ChromaDB database** by mounting the `chroma_db/` directory to a persistent volume if running in a container.