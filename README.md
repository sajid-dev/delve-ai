# delve-ai

A production‑grade backend for a chat application powered by large language models.  It exposes a REST API using **FastAPI** and uses **LangChain** to manage communication with your chosen model via a persistent **ChromaDB** vector store.  Structured logging is handled via **Loguru**.

## Features

- **Chat endpoint** – `POST /chat` accepts a prompt and returns an AI response.
- **Health check** – `GET /health` responds with the current health of the service.
  - **LangChain integration** – uses `langchain` and `langchain‑openai` for conversation chains backed by a `ChromaDB` vector store.
- **Structured logging** – uses Loguru with log rotation and retention to persist logs in `logs/app.log`.
- **Configuration via environment variables** – secrets and tunables are loaded from a `.env` file via Pydantic settings.

## Getting started

1. **Clone the repository** and change into the backend directory:

   ```bash
   git clone <your‑repo>
   cd delve-ai
   ```

2. **Install uv and set up your project environment**.

   This project is designed to work seamlessly with the [uv](https://github.com/astral-sh/uv) Python project manager.  If you don't already have uv installed, you can install it with pip:

   ```bash
   pip install uv
   ```

   Once uv is installed, run the following command in the project root to create a `.venv` directory and install all dependencies defined in `pyproject.toml`:

   ```bash
   uv sync
   ```

   The first time you run `uv sync` it will create a virtual environment in `.venv/` and generate a `uv.lock` lockfile.  If you prefer to use a traditional virtual environment without uv, you can still create one manually:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install .
   ```

3. **Configure environment variables**.  Copy `.env.example` to `.env` and fill in your language model credentials and settings.  At a minimum you must provide `LLM_API_KEY`.  You can optionally override `LLM_BASE_URL`, `LLM_MODEL` and other tuning parameters:

   ```bash
   cp .env.example .env
   # Then edit .env and set LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, etc.
   ```

4. **Run the server**.  If you're using uv, you can run the FastAPI application directly in the uv-managed environment:

   ```bash
   uv run -- uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
   ```

   Alternatively, if you activated a virtual environment manually, run:

   ```bash
   uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Test the API**.  Use `curl` or a REST client to send a chat request:

   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message":"Hello"}'
   ```

   and check the health endpoint:

   ```bash
   curl http://localhost:8000/health
   ```

## Project structure

```
delve-ai/
├── chroma_db/          # Persistent vector store for ChromaDB
├── .python-version     # Python version pin for uv-managed environments
├── logs/               # Log files written by Loguru
├── src/                # Source code organised by domain
│   ├── main.py         # FastAPI application entry point
│   ├── config/         # Configuration classes including settings.py, app_config.py, llm_config.py, logging_config.py
│   ├── controllers/    # API controllers
│   ├── services/       # Business logic
│   ├── models/         # Pydantic models for requests and responses
│   ├── memory/         # Conversation memory management
│   └── utils/          # Helper functions, error handling and logging
```

## Notes

* This backend uses `VectorStoreRetrieverMemory` from LangChain to store conversation history in a persistent `ChromaDB` database.  The database persists under the `chroma_db/` directory so that previous messages are used to provide context to subsequent prompts.
* Make sure the `LLM_API_KEY` environment variable is set; without it, the service will return an error.  If you are using a provider other than OpenAI, also set `LLM_BASE_URL` and the appropriate `LLM_MODEL`.

## API endpoints

The backend exposes a small number of HTTP endpoints via FastAPI.  All responses are JSON encoded.

### `GET /health`

Returns a simple health check indicating the service is running.

**Request:**

```
GET /health
```

**Response:**

```json
{
  "status": "ok"
}
```

### `POST /chat`

Accepts a user message and returns an assistant response.  The request body must include a non‑empty `message` field.  Input validation is handled by Pydantic; if the `message` field is missing, empty or exceeds 500 characters, FastAPI will return a 422 error.

**Request:**

```json
{
  "message": "Hello, who are you?"
}
```

**Response (200):**

```json
{
  "answer": "Hello! I'm your AI assistant. How can I help you today?"
}
```

If the LLM or memory backend fails, the service returns a 500 Internal Server Error with a JSON body such as:

```json
{
  "detail": "LLM generation failed"
}
```