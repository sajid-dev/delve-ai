# delve-ai

A production‑grade backend for a chat application powered by large language models.  It exposes a REST API using **FastAPI** and uses **LangChain** to manage communication with your chosen model via a persistent **ChromaDB** vector store.  Structured logging is handled via **Loguru**.

## Features

The **delve‑ai** backend is designed as a production‑ready foundation for enterprise chat applications.  In addition to the basic chat and health endpoints, the architecture provides a rich feature set across multiple domains.  Some capabilities are fully implemented today, while others are designed into the architecture and can be enabled with additional configuration or plugins.

### Core API endpoints

- **Chat endpoint** – `POST /chat` accepts a prompt and returns an AI response.
- **Admin dashboard** – `GET /admin/dashboard` reports per-user conversations with aggregated token usage.
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
- **Conversation deletion** – individual conversation removal (*ready*).
- **Bulk operations** – clear all user history (*ready*).

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

   Admin operators can inspect usage analytics:

   ```bash
   curl http://localhost:8000/admin/dashboard
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

Starts or continues a conversation and returns an assistant response.  The request body must include a non-empty `message` field.  Optional `user_id` and `conversation_id` fields allow the client to specify an existing user and conversation.  If omitted, new identifiers are generated and returned in the response.  Input validation is handled by Pydantic; if the `message` field is missing, empty or exceeds 500 characters, FastAPI will return a 422 error.

When a `user_id` is present it is forwarded to the underlying language model, enabling per-user analytics, auditing or quota management at the model provider level.

**Request:**

```json
{
  "message": "Hello, who are you?",
  "user_id": "123e4567-e89b-12d3-a456-426614174000",        // optional
  "conversation_id": "9c5f869a-2b8a-47f6-9ffd-0cbbd9e02c66" // optional
}
```

**Response (200):**

```json
{
  "user_id": "123e4567-e89b-12d3-a456-426614174000",
  "conversation_id": "9c5f869a-2b8a-47f6-9ffd-0cbbd9e02c66",
  "answer": "Hello! I'm your AI assistant. How can I help you today?"
}
```

If the LLM or memory backend fails, the service returns a 500 Internal Server Error with a JSON body such as:

```json
{
  "detail": "LLM generation failed"
}
```

### `GET /conversations`

Lists all conversations for a given user.  The `user_id` query parameter is required.

**Request:**

```
GET /conversations?user_id=123e4567-e89b-12d3-a456-426614174000
```

**Response (200):**

```json
[
  {
    "conversation_id": "9c5f869a-2b8a-47f6-9ffd-0cbbd9e02c66",
    "user_id": "123e4567-e89b-12d3-a456-426614174000",
    "title": "Hello, who are you?",           // optional title derived from first message
    "created_at": "2025-09-17T10:00:00Z",
    "updated_at": "2025-09-17T10:01:00Z",
    "message_count": 2,
    "messages": [ /* omitted for brevity */ ]
  }
]
```

### `GET /conversations/{conversation_id}`

Retrieves a single conversation for a user.  Both `conversation_id` (path) and `user_id` (query) parameters are required.

**Request:**

```
GET /conversations/9c5f869a-2b8a-47f6-9ffd-0cbbd9e02c66?user_id=123e4567-e89b-12d3-a456-426614174000
```

**Response (200):**

```json
{
  "conversation_id": "9c5f869a-2b8a-47f6-9ffd-0cbbd9e02c66",
  "user_id": "123e4567-e89b-12d3-a456-426614174000",
  "title": "Hello, who are you?",
  "created_at": "2025-09-17T10:00:00Z",
  "updated_at": "2025-09-17T10:01:00Z",
  "message_count": 2,
  "messages": [
    { "role": "user", "content": "Hello, who are you?", "timestamp": "2025-09-17T10:00:00Z" },
    { "role": "assistant", "content": "Hello! I'm your AI assistant. How can I help you today?", "timestamp": "2025-09-17T10:01:00Z" }
  ]
}
```

If the conversation does not exist, the service returns a 404 Not Found.

### `DELETE /conversations/{conversation_id}`

Deletes a specific conversation for a user.  Requires both the `conversation_id` path parameter and the `user_id` query parameter.  Returns a 204 No Content response on success.

### `DELETE /conversations`

Deletes all conversations for a user.  Requires the `user_id` query parameter.  Returns a 204 No Content response on success.
