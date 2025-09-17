# API Documentation

This document describes the available REST endpoints for the LLM Chat backend.

## `POST /chat`

Accepts a JSON body with a single field `message` containing the user's prompt.

**Request**

```json
{
  "message": "Hello, world!"
}
```

**Response**

```json
{
  "answer": "Hi there! How can I assist you today?"
}
```

## `GET /health`

Returns the health status of the service.

**Response**

```json
{
  "status": "ok"
}
```