# delve-ai

A production‑grade backend for a chat application powered by large language models.  It exposes a REST API using **FastAPI** and uses **LangChain** to manage communication with your chosen model via a persistent **ChromaDB** vector store.  Structured logging is handled via **Loguru**.

## Features

The **delve‑ai** backend is designed as a production‑ready foundation for enterprise chat applications.  In addition to the basic chat and health endpoints, the architecture provides a rich feature set across multiple domains.  Some capabilities are fully implemented today, while others are designed into the architecture and can be enabled with additional configuration or plugins.

### Core API endpoints

- **Chat endpoint** – `POST /chat` accepts a prompt and returns an AI response.
- **Admin dashboard** – `GET /admin/dashboard` reports per-user conversations with token counts, user totals, and active-user metrics.
- **Health check** – `GET /health` responds with the current health of the service.
- **LangChain integration** – uses `langchain` and `langchain‑openai` for conversation chains backed by a `ChromaDB` vector store.
- **Structured logging** – uses Loguru with log rotation and retention to persist logs in `logs/app.log`.
- **Configuration via environment variables** – secrets and tunables are loaded from a `.env` file via Pydantic settings.

### Extended capabilities

Below is an overview of additional features supported by the architecture.  Items marked *ready* are planned and the design accommodates them, but they may require further implementation or configuration to activate.

#### 💾 Memory & Storage

- **ChromaDB integration** – persistent vector database for conversation storage (implemented).
- **User‑based isolation** – complete separation of user conversations (*ready*).
- **Conversation persistence** – automatic saving of chat history (implemented).
- **Vector similarity search** – semantic search through past conversations (implemented via `Chroma` retriever).
- **Multiple storage back‑ends** – architecture supports alternative stores such as Redis or PostgreSQL (*ready*).
- **Automatic data persistence** – no manual saving required (implemented).

#### 🔄 Conversation management

- **Multiple conversations** – ability for users to have multiple simultaneous chats (*ready*; requires a user/memory manager).
- **Conversation history** – full message history retrieval (implemented).
- **Context‑aware responses** – uses previous conversation context (implemented).
- **Conversation metadata** – titles, timestamps, message counts (*ready*).
- **Conversation deletion** – individual conversation removal (implemented).
- **Bulk operations** – clear all user history (implemented via admin API).

#### 👥 User management

- **Automatic user ID generation** – unique user identification (*ready*).
- **Session persistence** – maintains state across sessions (implemented via persistent storage).
- **User data isolation** – complete privacy between users (*ready*; requires user memory manager).
- **Provider-aware attribution** – forwards the supplied `user_id` to the LLM for auditing and usage tracking (implemented).
- **User preference storage** – architecture for user settings (*ready*).
- **Authentication ready** – prepared for user login systems (*ready*).

#### ⚙️ Configuration & settings

- **Environment‑based configuration** – `.env` file support (implemented).
- **Runtime configuration** – hot‑reload capable settings (*ready*).
- **Multiple environment support** – dev, staging, production (implemented).
- **Centralised configuration** – single source for all settings (implemented via `src/config`).
- **Validation** – type‑safe configuration management using Pydantic (implemented).

#### 🛡️ Error handling & reliability

- **Comprehensive error handling** – graceful degradation on failures (implemented via custom exceptions).
- **Automatic retry logic** – retry failed API calls (*ready*; would be implemented at the API client layer).
- **Circuit breaker pattern** – prevents cascading failures (*ready*).
- **Fallback mechanisms** – continue working during partial outages (*ready*).
- **Detailed error logging** – complete error context for debugging (implemented via Loguru).
- **User‑friendly error messages** – appropriate feedback for users (implemented).

#### 📊 Monitoring & analytics

- **Service health checks** – comprehensive system monitoring (implemented).
- **Conversation analytics dashboard** – built-in admin endpoint summarising usage across users (implemented).
- **Performance metrics** – response times, success rates (*ready*).
- **Usage statistics** – conversation counts, message volumes (*ready*).
- **Logging infrastructure** – structured logging with multiple levels (implemented).
- **Audit trail** – complete operation logging (*ready*).

#### 🔧 API & integration

- **REST API ready** – full API endpoint architecture (implemented).
- **Webhook support ready** – prepared for external integrations (*ready*).
- **Batch processing support** – bulk operations capability (*ready*).
- **Custom prompt engineering** – support for specialised prompts (implemented via LangChain chains).
- **Multi‑modal ready** – architecture prepared for images/audio (*ready*).

#### 🚀 Performance & scalability

- **Caching mechanisms** – performance optimisation (*ready*).
- **Connection pooling** – efficient resource usage (*ready*; depends on HTTP client).
- **Horizontal scaling ready** – stateless service design (implemented).
- **Load balancer compatible** – production deployment ready (implemented).
- **Resource optimisation** – efficient memory and CPU usage (implemented via asynchronous FastAPI design).

#### 🧪 Development & testing

- **Comprehensive test suite** – unit, integration, end‑to‑end tests (*ready*; test scaffolding can be added).
- **Type safety** – full type annotations throughout (implemented).
- **Debug mode** – development debugging support (implemented via `app_debug`).
- **Mock services** – testing without external dependencies (*ready*).
- **Documentation ready** – comprehensive code documentation (implemented in `src/docs`).

#### 🔒 Security & compliance

- **API key security** – environment variable protection (implemented; secrets loaded via `.env`).
- **Input validation** – SQL injection and XSS prevention (*ready*; depending on storage implementation).
- **Data encryption ready** – prepared for encrypted storage (*ready*).
- **Privacy by design** – user data isolation built‑in (*ready*).
- **Audit logging** – compliance and security auditing (*ready*).

#### 🌐 Deployment & operations

- **Docker ready** – containerisation support (*ready*; Dockerfile can be added for container builds).
- **Cloud platform compatible** – AWS, GCP, Azure (implemented; FastAPI is platform‑agnostic).
- **Streamlit sharing ready** – one‑click deployment (*ready*).
- **Traditional server support** – WSGI/ASGI compatible (implemented via Uvicorn/ASGI).
- **Zero‑downtime deployment ready** – blue‑green deployment capable (*ready*; requires deployment orchestration).

#### 🔄 Workflow features

- **End‑to‑end processing** – complete message handling pipeline (implemented).
- **Memory integration** – automatic context management (implemented).
- **Response generation** – AI response creation and formatting (implemented via LangChain).
- **Continuous conversation** – multi‑turn dialogue support (implemented).
- **Context window management** – smart context truncation (*ready*).

#### 📈 Advanced features

- **Conversation summarisation ready** – architecture for summary generation (*ready*).
- **Knowledge base integration ready** – prepared for external data sources (*ready*).
- **Custom embedding support** – multiple embedding model compatibility (*ready*; via `OpenAIEmbeddings` parameters).
- **Semantic search** – intelligent conversation retrieval (implemented via vector similarity search).
- **Personalisation ready** – user‑specific behaviour adaptation (*ready*).

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

   To expose Model Context Protocol (MCP) tools to the assistant, set
   `MCP_ENABLED=true` and populate `MCP_SERVERS` with a JSON array of server
   definitions. Each entry supports `command`, `args`, `env`, `cwd`, optional
   per-server `transport`, and optional `trigger_keywords` for per-server
   routing. When using HTTP transports you can also provide `url`, `headers`,
   and timeout values (expressed in seconds). For example, to enable the
   open-source Shadcn UI MCP server alongside a custom toolchain:

   ```bash
   MCP_ENABLED=true
   MCP_SERVERS='[
     {"name":"analysis","command":"/opt/tools/mcp-analytics","args":["--mode","batch"]},
     {"name":"shadcn-ui","command":"npx","args":["@modelcontextprotocol/server-shadcn"],
      "trigger_keywords":["ui","component","shadcn"]}
   ]'
   ```

   If you omit `MCP_SERVERS` entirely the backend will fall back to the default
   definitions in `src/config/mcp_servers.py`, which makes local development
   easier—edit that file to tweak built-in servers.  The legacy
   `LLM_MCP_ENABLED`, `LLM_MCP_TRANSPORT`, `LLM_MCP_TRIGGER_KEYWORDS` and
   `LLM_MCP_SERVERS` variables are still honoured for backward compatibility,
   but new deployments should prefer the streamlined `MCP_*` names.
   When relying on the legacy single-server variables you can also supply HTTP
   options such as `MCP_SERVER_URL`, `MCP_SERVER_HEADERS`,
   `MCP_SERVER_TIMEOUT`, `MCP_SERVER_SSE_READ_TIMEOUT`, and
   `MCP_SERVER_TERMINATE_ON_CLOSE` (or their `LLM_` equivalents) to configure
   remote MCP transports without switching to JSON.

   The backend supports the full suite of transports exposed by
   `langchain-mcp-adapters`, including `stdio`, `sse`, and streamable HTTP.
   Set the global `MCP_TRANSPORT` (or legacy `LLM_MCP_TRANSPORT`) to the desired
   value, or override the transport on individual server definitions. Under the
   hood the backend relies on LangChain's `langchain-mcp` toolkit together with
   `langchain-mcp-adapters` to negotiate and multiplex MCP sessions, so make
   sure both packages are installed when running outside the provided virtual
   environment.

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
     -d '{"message":"Hello","session_id":"demo-session-1"}'
   ```

   and check the health endpoint:

   ```bash
   curl http://localhost:8000/health
   ```

   Admin operators can inspect usage analytics, review user sessions, or clear stored sessions:

   ```bash
   curl http://localhost:8000/admin/dashboard
   curl http://localhost:8000/admin/conversations/<user_id>
   curl -X DELETE http://localhost:8000/admin/conversations
   curl -X DELETE http://localhost:8000/admin/conversations/<user_id>
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

## API services

The FastAPI application exposes a small, well-defined surface area.  Every endpoint returns JSON unless stated otherwise.

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Lightweight health probe used by load balancers and uptime checks. |
| POST | `/chat` | Submit a user message and receive the assistant response. |
| GET | `/sessions` | List every saved session for a specific user. |
| GET | `/sessions/{session_id}` | Retrieve the full transcript of a single session. |
| DELETE | `/sessions/{session_id}` | Remove one session for the current user. |
| DELETE | `/sessions` | Remove all sessions for the current user. |
| GET | `/admin/dashboard` | Aggregate analytics across all users and sessions. |
| GET | `/admin/conversations/{user_id}` | Inspect every session stored for the given user. |
| DELETE | `/admin/conversations` | Purge all conversations across every user. |
| DELETE | `/admin/conversations/{user_id}` | Purge every conversation that belongs to one user. |

### Health

#### `GET /health`

Returns the operational status of the service.

**Request**

```
GET /health
```

**Response 200**

```json
{
  "status": "ok"
}
```

### Chat

#### `POST /chat`

Creates or continues a conversation.  Supply `user_id` and `session_id` to resume an existing thread; omit them to let the service create new IDs.

**Request**

```json
{
  "message": "Hello, who are you?",
  "user_id": "123e4567-e89b-12d3-a456-426614174000",
  "session_id": "9c5f869a-2b8a-47f6-9ffd-0cbbd9e02c66"
}
```

**Response 200**

```json
{
  "user_id": "123e4567-e89b-12d3-a456-426614174000",
  "session_id": "9c5f869a-2b8a-47f6-9ffd-0cbbd9e02c66",
  "data": {
    "components": [
      {
        "type": "text",
        "payload": {
          "content": "Hello! I'm your AI assistant. How can I help you today?"
        }
      }
    ]
  }
}
```

`data.components` describes how front-ends should render the answer.  Additional component types (`list`, `table`, `chart`, `code`, `image`, `json`, `markdown`) are emitted when the LLM response includes structured content.

**Response 500 (example)**

```json
{
  "detail": "LLM processing failed"
}
```

### Session management

#### `GET /sessions`

Lists every session for a user.  The `user_id` query parameter is required.

**Request**

```
GET /sessions?user_id=123e4567-e89b-12d3-a456-426614174000
```

**Response 200**

```json
[
  {
    "session_id": "9c5f869a-2b8a-47f6-9ffd-0cbbd9e02c66",
    "user_id": "123e4567-e89b-12d3-a456-426614174000",
    "title": "Hello, who are you?",
    "created_at": "2025-09-17T10:00:00Z",
    "updated_at": "2025-09-17T10:01:00Z",
    "message_count": 2,
    "messages": []
  }
]
```

#### `GET /sessions/{session_id}`

Retrieves an entire conversation for the requesting user.  Requires both the `session_id` path parameter and the `user_id` query parameter.

**Request**

```
GET /sessions/9c5f869a-2b8a-47f6-9ffd-0cbbd9e02c66?user_id=123e4567-e89b-12d3-a456-426614174000
```

**Response 200**

```json
{
  "session_id": "9c5f869a-2b8a-47f6-9ffd-0cbbd9e02c66",
  "user_id": "123e4567-e89b-12d3-a456-426614174000",
  "title": "Hello, who are you?",
  "created_at": "2025-09-17T10:00:00Z",
  "updated_at": "2025-09-17T10:01:00Z",
  "message_count": 2,
  "messages": [
    {
      "role": "user",
      "content": "Hello, who are you?",
      "content_type": "text",
      "timestamp": "2025-09-17T10:00:00Z",
      "components": null
    },
    {
      "role": "assistant",
      "content": "Hello! I'm your AI assistant. How can I help you today?",
      "content_type": "text",
      "timestamp": "2025-09-17T10:01:00Z",
      "components": [
        {
          "type": "text",
          "payload": {
            "content": "Hello! I'm your AI assistant. How can I help you today?"
          }
        }
      ]
    }
  ]
}
```

**Response 404 (example)**

```json
{
  "detail": "Session not found"
}
```

#### `DELETE /sessions/{session_id}`

Deletes a single conversation for the requesting user.  Requires both `session_id` and `user_id`.

**Request**

```
DELETE /sessions/9c5f869a-2b8a-47f6-9ffd-0cbbd9e02c66?user_id=123e4567-e89b-12d3-a456-426614174000
```

**Response 204**

No body.

#### `DELETE /sessions`

Deletes every conversation that belongs to the requesting user.

**Request**

```
DELETE /sessions?user_id=123e4567-e89b-12d3-a456-426614174000
```

**Response 204**

No body.

### Admin operations

Admin endpoints operate across all users and should be protected behind authentication in production deployments.

#### `GET /admin/dashboard`

Returns aggregate usage metrics across every stored session.

**Request**

```
GET /admin/dashboard
```

**Response 200**

```json
{
  "total_users": 3,
  "active_users": 2,
  "total_sessions": 12,
  "total_tokens": 5475,
  "users": [
    {
      "user_id": "123e4567-e89b-12d3-a456-426614174000",
      "session_count": 4,
      "total_tokens": 2200,
      "last_active": "2025-09-19T08:30:00Z",
      "is_active": true,
      "sessions": [
        {
          "session_id": "9c5f869a-2b8a-47f6-9ffd-0cbbd9e02c66",
          "title": "Hello, who are you?",
          "message_count": 6,
          "created_at": "2025-09-17T10:00:00Z",
          "updated_at": "2025-09-19T08:30:00Z",
          "tokens_used": 640,
          "latest_answer": {
            "session_id": "9c5f869a-2b8a-47f6-9ffd-0cbbd9e02c66",
            "timestamp": "2025-09-19T08:30:00Z"
          }
        }
      ]
    }
  ]
}
```

#### `GET /admin/conversations/{user_id}`

Returns every saved session for the specified user, including full transcripts.

**Request**

```
GET /admin/conversations/123e4567-e89b-12d3-a456-426614174000
```

**Response 200**

```json
[
  {
    "session_id": "9c5f869a-2b8a-47f6-9ffd-0cbbd9e02c66",
    "user_id": "123e4567-e89b-12d3-a456-426614174000",
    "title": "Hello, who are you?",
    "created_at": "2025-09-17T10:00:00Z",
    "updated_at": "2025-09-17T10:01:00Z",
    "message_count": 2,
    "messages": [
      {
        "role": "user",
        "content": "Hello, who are you?",
        "content_type": "text",
        "timestamp": "2025-09-17T10:00:00Z",
        "components": null
      },
      {
        "role": "assistant",
        "content": "Hello! I'm your AI assistant. How can I help you today?",
        "content_type": "text",
        "timestamp": "2025-09-17T10:01:00Z",
        "components": [
          {
            "type": "text",
            "payload": {
              "content": "Hello! I'm your AI assistant. How can I help you today?"
            }
          }
        ]
      }
    ]
  }
]
```

#### `DELETE /admin/conversations`

Clears every stored session across all users.  Use with caution.

**Request**

```
DELETE /admin/conversations
```

**Response 200**

```json
{
  "status": "ok"
}
```

#### `DELETE /admin/conversations/{user_id}`

Clears every session belonging to the given user without affecting others.

**Request**

```
DELETE /admin/conversations/123e4567-e89b-12d3-a456-426614174000
```

**Response 200**

```json
{
  "status": "ok"
}
```
