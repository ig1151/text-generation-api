from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.responses import JSONResponse as StarletteJSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Literal
import os
import time
import uuid
import random
import asyncio
import logging
import json
import sqlite3
import threading
import secrets
from datetime import datetime

import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
APP_VERSION         = os.getenv("APP_VERSION", "1.0.0")
ENVIRONMENT         = os.getenv("ENVIRONMENT", "production")

OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "").strip() or None
OPENAI_BASE_URL     = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_TEXT_MODEL   = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o")

ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY", "").strip() or None
ANTHROPIC_BASE_URL  = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
ANTHROPIC_TEXT_MODEL = os.getenv("ANTHROPIC_TEXT_MODEL", "claude-sonnet-4-5")

ADMIN_KEY           = os.getenv("ADMIN_KEY", "").strip() or None
REQUIRE_API_KEY     = os.getenv("REQUIRE_API_KEY", "true").lower() == "true"

HTTP_TIMEOUT        = float(os.getenv("HTTP_TIMEOUT_SECONDS", "30"))
MAX_RETRIES         = int(os.getenv("UPSTREAM_MAX_RETRIES", "3"))
BACKOFF_BASE        = float(os.getenv("UPSTREAM_BACKOFF_BASE_SECONDS", "0.5"))
BACKOFF_MAX         = float(os.getenv("UPSTREAM_BACKOFF_MAX_SECONDS", "4.0"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("text-generation-api")

# ---------------------------------------------------------------------------
# SQLite key store
# ---------------------------------------------------------------------------
DB_PATH   = os.getenv("DB_PATH", "text_api_keys.db")
_db_lock  = threading.Lock()

def _conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    with _db_lock:
        c = _conn()
        c.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                key        TEXT PRIMARY KEY,
                label      TEXT NOT NULL,
                active     INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            )
        """)
        c.commit(); c.close()

def db_add_key(key: str, label: str):
    with _db_lock:
        c = _conn()
        c.execute("INSERT INTO api_keys (key, label, active, created_at) VALUES (?,?,1,?)",
                  (key, label, datetime.utcnow().isoformat()))
        c.commit(); c.close()

def db_valid(key: str) -> bool:
    with _db_lock:
        c = _conn()
        row = c.execute("SELECT 1 FROM api_keys WHERE key=? AND active=1", (key,)).fetchone()
        c.close()
        return row is not None

def db_revoke(key: str) -> bool:
    with _db_lock:
        c = _conn()
        cur = c.execute("UPDATE api_keys SET active=0 WHERE key=?", (key,))
        c.commit(); c.close()
        return cur.rowcount > 0

def db_list() -> list:
    with _db_lock:
        c = _conn()
        rows = c.execute("SELECT key, label, active, created_at FROM api_keys ORDER BY created_at DESC").fetchall()
        c.close()
        return [dict(r) for r in rows]

init_db()

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

app = FastAPI(
    title="Text Generation API",
    version=APP_VERSION,
    description="Multi-provider text generation with automatic Anthropic/OpenAI fallback.",
)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return StarletteJSONResponse(status_code=429, content={"error": "Rate limit exceeded. Max 60 requests/minute per IP."})

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
api_key_header   = APIKeyHeader(name="X-API-Key",   auto_error=False)
admin_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)

async def require_api_key(key: str = Depends(api_key_header)):
    if not REQUIRE_API_KEY:
        return
    if not key or not db_valid(key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key. Pass it as X-API-Key header.")

async def require_admin_key(key: str = Depends(admin_key_header)):
    if not ADMIN_KEY:
        raise HTTPException(status_code=503, detail="Admin key not configured on this server.")
    if not key or key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing admin key. Pass it as X-Admin-Key header.")

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class TextRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=50000)
    system: Optional[str] = Field(default=None, max_length=10000)
    model: Optional[str] = None
    max_output_tokens: int = Field(default=512, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0, le=2)
    preferred_provider: Literal["auto", "openai", "anthropic"] = "auto"

class TextResponse(BaseModel):
    text: str
    provider: str
    model: str
    request_id: str

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _backoff(attempt: int) -> float:
    exp = BACKOFF_BASE * (2 ** (attempt - 1))
    return min(BACKOFF_MAX, exp + random.uniform(0, min(0.25, exp / 4)))

def _retryable(code: int) -> bool:
    return code in {408, 425, 429, 500, 502, 503, 504}

async def _post(client: httpx.AsyncClient, provider: str, url: str, headers: dict, body: dict) -> dict:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = await client.request("POST", url, headers=headers, json=body)
            if r.status_code >= 400:
                if _retryable(r.status_code) and attempt < MAX_RETRIES:
                    await asyncio.sleep(_backoff(attempt)); continue
                raise HTTPException(status_code=r.status_code, detail=f"Upstream error from {provider}: {r.text[:300]}")
            return r.json()
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            last_err = e
            if attempt >= MAX_RETRIES:
                raise HTTPException(status_code=504, detail=f"Timeout from {provider}")
            await asyncio.sleep(_backoff(attempt))
    raise HTTPException(status_code=502, detail=f"Failed to reach {provider}") from last_err

# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
async def _openai_text(client: httpx.AsyncClient, payload: TextRequest) -> TextResponse:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured")
    model = payload.model or OPENAI_TEXT_MODEL
    body = {"model": model, "input": payload.prompt, "max_output_tokens": payload.max_output_tokens, "temperature": payload.temperature}
    if payload.system:
        body["instructions"] = payload.system
    data = await _post(client, "openai", f"{OPENAI_BASE_URL}/responses",
                       {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}, body)
    text = data.get("output_text", "")
    if not text:
        for item in data.get("output", []):
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        text += c.get("text", "")
    return TextResponse(text=text, provider="openai", model=model, request_id=str(uuid.uuid4()))

async def _anthropic_text(client: httpx.AsyncClient, payload: TextRequest) -> TextResponse:
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is not configured")
    model = payload.model or ANTHROPIC_TEXT_MODEL
    body = {"model": model, "max_tokens": payload.max_output_tokens, "temperature": payload.temperature,
            "messages": [{"role": "user", "content": payload.prompt}]}
    if payload.system:
        body["system"] = payload.system
    data = await _post(client, "anthropic", f"{ANTHROPIC_BASE_URL}/messages",
                       {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}, body)
    text = "".join(p.get("text", "") for p in data.get("content", []) if p.get("type") == "text")
    return TextResponse(text=text, provider="anthropic", model=model, request_id=str(uuid.uuid4()))

async def _generate_with_fallback(client: httpx.AsyncClient, payload: TextRequest) -> TextResponse:
    if payload.preferred_provider == "anthropic":
        providers = [("anthropic", lambda: _anthropic_text(client, payload)),
                     ("openai",    lambda: _openai_text(client, payload))]
    else:
        providers = [("openai",    lambda: _openai_text(client, payload)),
                     ("anthropic", lambda: _anthropic_text(client, payload))]
    first_err = None
    for name, op in providers:
        try:
            return await op()
        except HTTPException as e:
            logger.warning("Provider failed provider=%s detail=%s", name, e.detail)
            if first_err is None: first_err = e
    raise first_err or HTTPException(status_code=502, detail="No text provider available")

# ---------------------------------------------------------------------------
# Key management
# ---------------------------------------------------------------------------
@app.post("/v1/keys/generate")
async def generate_key(label: str, _: None = Depends(require_admin_key)):
    key = "txt-" + secrets.token_urlsafe(32)
    db_add_key(key, label)
    return {"label": label, "api_key": key, "note": "Store securely — cannot be retrieved again."}

@app.get("/v1/keys")
async def list_keys(_: None = Depends(require_admin_key)):
    keys = db_list()
    for k in keys: k["key"] = k["key"][:8] + "••••••••"
    return {"count": len(keys), "keys": keys}

@app.delete("/v1/keys/revoke")
async def revoke_key(key: str, _: None = Depends(require_admin_key)):
    if not db_revoke(key):
        raise HTTPException(status_code=404, detail="Key not found.")
    return {"revoked": True, "key": key[:8] + "••••••••"}

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "environment": ENVIRONMENT, "version": APP_VERSION,
            "providers": {"openai": bool(OPENAI_API_KEY), "anthropic": bool(ANTHROPIC_API_KEY)}}

@app.get("/ready")
async def ready():
    checks = {"openai_configured": bool(OPENAI_API_KEY),
              "anthropic_configured": bool(ANTHROPIC_API_KEY),
              "at_least_one_provider": bool(OPENAI_API_KEY or ANTHROPIC_API_KEY)}
    return {"ready": all(checks.values()), "checks": checks}

@app.post("/v1/text/generate", response_model=TextResponse)
@limiter.limit("60/minute")
async def text_generate(
    request: Request,
    payload: TextRequest,
    _: None = Depends(require_api_key),
):
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        return await _generate_with_fallback(client, payload)
