#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

MODELS = [
    {"name": "93-Feature", "url": "https://wingo-api-two.vercel.app", "short": "M1"},
    {"name": "10M V2", "url": "https://wingo-10m-api-akashyt12s-projects.vercel.app", "short": "M2"},
    {"name": "Pattern-5M", "url": "https://wingo-pattern-api-akashyt12s-projects.vercel.app", "short": "M3"},
]
HISTORY_URL = "https://wingo-api-two.vercel.app/history"
HISTORY_LIMIT = int(os.environ.get("HISTORY_LIMIT", "10"))
POLL_SECONDS = max(0.5, float(os.environ.get("POLL_SECONDS", "2")))
HISTORY_TIMEOUT = float(os.environ.get("HISTORY_TIMEOUT", "3"))
MODEL_TIMEOUT = float(os.environ.get("MODEL_TIMEOUT", "5"))
HISTORY_FILE = Path(os.environ.get("BS_HISTORY_FILE", str(Path(__file__).with_name("big_small_history.json"))))
MAX_HISTORY = int(os.environ.get("MAX_HISTORY", "500"))

app = FastAPI(title="Stylo Final BIG/SMALL API", docs_url=None, redoc_url=None)

_state: dict[str, Any] = {
    "result": None,
    "last_poll": 0.0,
    "poll_count": 0,
    "latency_ms": None,
    "error": None,
    "seen_period": None,
}
_history: list[dict[str, Any]] = []
_sse: set[asyncio.Queue] = set()


def _load_history() -> None:
    try:
        if HISTORY_FILE.exists():
            data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                _history.extend(data[-MAX_HISTORY:])
    except Exception:
        pass


def _save_history() -> None:
    try:
        HISTORY_FILE.write_text(json.dumps(_history, indent=2), encoding="utf-8")
    except Exception:
        pass


_load_history()


def _get(url: str, timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "stylo-final-big-small/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


async def _aget(url: str, timeout: float) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _get, url, timeout)


def _view(meta: dict[str, str], data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {"short": meta["short"], "name": meta["name"], "size": "-", "p_big": 0.5, "confidence": 0.0, "period": "", "error": "invalid response"}
    p = data.get("prediction") if isinstance(data.get("prediction"), dict) else {}
    pr = p.get("probabilities") if isinstance(p.get("probabilities"), dict) else {}
    if not pr and isinstance(data.get("probabilities"), dict):
        pr = data["probabilities"]
    raw_size = str(p.get("size") or p.get("big_small") or data.get("size") or "-").upper()
    size = raw_size if raw_size in ("BIG", "SMALL") else "-"
    try:
        p_big = float(pr.get("big", data.get("p_big", 0.5)))
    except (TypeError, ValueError):
        p_big = 0.5
    out = {
        "short": meta["short"],
        "name": meta["name"],
        "size": size,
        "p_big": round(p_big, 6),
        "confidence": round(abs(p_big - 0.5), 6),
        "period": str(data.get("period") or p.get("period") or ""),
    }
    if size == "-":
        out["error"] = "no big/small in response"
    return out


def _draw_size(draw: dict[str, Any]) -> str | None:
    size = str(draw.get("size", "")).upper()
    if size in ("BIG", "SMALL"):
        return size
    for key in ("number", "digit", "result", "value"):
        value = draw.get(key)
        if value is not None and str(value).isdigit():
            return "BIG" if int(value) >= 5 else "SMALL"
    return None


def _resolve(history: Any) -> None:
    if not isinstance(history, dict):
        return
    draws = history.get("draws") or []
    actual = {str(d.get("period")): d for d in draws if isinstance(d, dict)}
    changed = False
    for rec in _history:
        if rec.get("status") != "pending":
            continue
        draw = actual.get(str(rec.get("period")))
        if not draw:
            continue
        size = _draw_size(draw)
        if not size:
            continue
        rec["actual_size"] = size
        rec["result"] = "WIN" if rec.get("final_size") == size else "LOSS"
        rec["status"] = "done"
        rec["resolved_at"] = datetime.now(timezone.utc).isoformat()
        changed = True
    if changed:
        _save_history()


def _stats() -> dict[str, Any]:
    done = [r for r in _history if r.get("status") == "done"]
    wins = sum(r.get("result") == "WIN" for r in done)
    streak = 0
    for rec in reversed(done):
        if rec.get("result") == "WIN":
            streak += 1
        else:
            break
    return {
        "total": len(done),
        "pending": len(_history) - len(done),
        "wins": wins,
        "losses": len(done) - wins,
        "accuracy": round(wins / len(done), 4) if done else None,
        "streak": streak,
        "last_result": done[-1].get("result") if done else None,
    }


def _add_prediction(result: dict[str, Any]) -> None:
    period = str(result["period"])
    if _history and str(_history[-1].get("period")) == period:
        return
    _history.append({
        "period": period,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "final_size": result["size"],
        "confidence": result["confidence"],
        "agreement": result["agreement"],
        "votes": {v["short"]: v["size"] for v in result["models"]},
        "status": "pending",
    })
    if len(_history) > MAX_HISTORY:
        del _history[:-MAX_HISTORY]
    _save_history()


def _build_result(preds: list[Any], history: Any) -> dict[str, Any]:
    views = [_view(meta, pred) for meta, pred in zip(MODELS, preds)]
    usable = [v for v in views if v["size"] in ("BIG", "SMALL")]
    if not usable:
        raise RuntimeError("all model APIs failed")
    confs: dict[str, float] = {}
    for view in usable:
        size = view["size"]
        confs[size] = max(confs.get(size, 0.0), view["confidence"])
    counts = Counter(v["size"] for v in usable)
    ranked = counts.most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        tied = {key for key, count in ranked if count == ranked[0][1]}
        final_size = max(tied, key=lambda key: confs.get(key, 0.0))
    else:
        final_size = ranked[0][0]
    periods = [int(v["period"]) for v in usable if str(v["period"]).isdigit()]
    period = str(max(periods)) if periods else datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S999")
    confidence = round(confs.get(final_size, 0.0), 6)
    result = {
        "status": "ok",
        "period": period,
        "size": final_size,
        "big_small": final_size,
        "prediction": {"size": final_size, "confidence": confidence},
        "confidence": confidence,
        "avg_p_big": round(sum(v["p_big"] for v in usable) / len(usable), 6),
        "agreement": f"{counts[final_size]}/{len(usable)}",
        "models": views,
        "stats": _stats(),
        "poll": {
            "count": _state["poll_count"] + 1,
            "interval_s": POLL_SECONDS,
            "latency_ms": _state["latency_ms"],
            "last_poll": _state["last_poll"],
            "error": None,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _resolve(history)
    _add_prediction(result)
    result["stats"] = _stats()
    return result


async def _poll_once() -> dict[str, Any]:
    started = time.perf_counter()
    error = None
    history: Any = {}
    try:
        history = await _aget(f"{HISTORY_URL}?limit={HISTORY_LIMIT}", HISTORY_TIMEOUT)
    except Exception as exc:
        error = f"history: {exc}"[:200]
    draws = history.get("draws") or [] if isinstance(history, dict) else []
    latest = ""
    if draws and isinstance(draws[-1], dict) and draws[-1].get("period"):
        latest = str(draws[-1]["period"])
    result = _state["result"]
    if latest and latest != _state["seen_period"]:
        urls = [m["url"] + ("/predict" if m["short"] == "M1" else "/predict?fresh=true") for m in MODELS]
        responses = await asyncio.gather(*[_aget(url, MODEL_TIMEOUT) for url in urls], return_exceptions=True)
        preds = [r if isinstance(r, dict) else {} for r in responses[:3]]
        try:
            result = _build_result(preds, history)
            _state["seen_period"] = latest
            _state["result"] = result
        except Exception as exc:
            error = f"models: {exc}"[:200]
            result = _state["result"]
    elif isinstance(history, dict):
        _resolve(history)
    if result is not None:
        result["stats"] = _stats()
    latency = round((time.perf_counter() - started) * 1000, 1)
    _state["last_poll"] = time.time()
    _state["poll_count"] += 1
    _state["latency_ms"] = latency
    _state["error"] = error
    if result is None:
        result = {"status": "error", "size": "WAIT", "period": latest, "error": error or "starting"}
    result["poll"] = {
        "count": _state["poll_count"],
        "interval_s": POLL_SECONDS,
        "latency_ms": latency,
        "last_poll": _state["last_poll"],
        "error": error,
    }
    for queue in list(_sse):
        try:
            queue.put_nowait(result)
        except asyncio.QueueFull:
            pass
    return result


async def _poller() -> None:
    while True:
        started = time.perf_counter()
        try:
            await _poll_once()
        except Exception as exc:
            _state["error"] = str(exc)[:200]
            _state["poll_count"] += 1
            _state["last_poll"] = time.time()
        await asyncio.sleep(max(0.25, POLL_SECONDS - (time.perf_counter() - started)))


@app.on_event("startup")
async def _startup() -> None:
    app.state.poller = asyncio.create_task(_poller())


@app.on_event("shutdown")
async def _shutdown() -> None:
    task = getattr(app.state, "poller", None)
    if task:
        task.cancel()


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "service": "Stylo Final BIG/SMALL API",
        "endpoints": ["/predict", "/size", "/health", "/stats", "/history", "/events", "/refresh"],
        "poll_seconds": POLL_SECONDS,
    }


@app.get("/predict")
async def predict(wait: bool = Query(False)) -> Any:
    if wait or _state["result"] is None:
        try:
            await _poll_once()
        except Exception as exc:
            return JSONResponse(status_code=502, content={"error": "poll failed", "detail": str(exc)[:200]})
    return _state["result"]


@app.get("/size")
async def only_size() -> PlainTextResponse:
    result = _state["result"]
    return PlainTextResponse(result["size"] if result else "WAIT")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok" if _state["result"] else "starting",
        "poll_count": _state["poll_count"],
        "last_poll": _state["last_poll"],
        "latency_ms": _state["latency_ms"],
        "error": _state["error"],
        "poll_seconds": POLL_SECONDS,
    }


@app.get("/stats")
async def stats() -> dict[str, Any]:
    return _stats()


@app.get("/history")
async def history(limit: int = Query(50, ge=1, le=500)) -> dict[str, Any]:
    return {"history": list(reversed(_history[-limit:]))}


@app.post("/refresh")
async def refresh() -> Any:
    try:
        await _poll_once()
    except Exception as exc:
        return JSONResponse(status_code=502, content={"error": "poll failed", "detail": str(exc)[:200]})
    return _state["result"]


@app.get("/events")
async def events() -> StreamingResponse:
    queue: asyncio.Queue = asyncio.Queue()
    _sse.add(queue)

    async def stream():
        try:
            if _state["result"]:
                yield f"data: {json.dumps(_state['result'], separators=(',', ':'))}\n\n"
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    continue
                yield f"data: {json.dumps(item, separators=(',', ':'))}\n\n"
        finally:
            _sse.discard(queue)

    return StreamingResponse(stream(), media_type="text/event-stream")
