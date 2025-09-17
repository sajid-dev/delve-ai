# Troubleshooting

This document lists common issues and their resolutions when running the LLM chat backend.

## Missing API key

If the service fails to start or throws a `ChatError` indicating that OpenAI authentication failed, ensure that the `OPENAI_API_KEY` environment variable is set in your `.env` file.

## Cannot write logs

If the application cannot write to `logs/app.log`, check that the `logs/` directory exists and has appropriate permissions.  The service will create the directory if it does not exist.

## No context in responses

If the assistant does not remember previous messages, verify that the `chroma_db/` directory is writable and that the `MemoryService` is not being reinitialised for each request.  The service uses an LRU cache to ensure a single instance.