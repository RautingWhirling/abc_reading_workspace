from __future__ import annotations

import json
import re
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .eval_hot_events import (
    _run_hot_event_evaluation,
    load_hot_events,
    select_hot_events,
)
from .image_event_loader import SUPPORTED_IMAGE_SUFFIXES, load_image_events


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENTS_PATH = PROJECT_ROOT / "eval" / "hot_event_opinion_variants_200.json"
WEB_RUNS_ROOT = PROJECT_ROOT / "outputs" / "web_runs"
FRONTEND_DIST = PROJECT_ROOT / "web" / "frontend" / "dist"
MAX_WORKERS = 1

_RUNS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()
_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_WORKERS)


class RunOptions(BaseModel):
    profile_limit: int | None = None
    max_selected_nodes: int = Field(default=5, ge=1, le=20)
    risk_level: Literal["low", "medium", "high"] | None = None
    campaign_window_hours: int = Field(default=24, ge=1, le=168)
    max_frequency_per_day: int = Field(default=3, ge=1, le=24)
    allowed_platforms: list[str] = Field(default_factory=lambda: ["weibo_simulated"])
    use_llm: bool = True
    event_limit: int | None = Field(default=None, ge=1)
    event_id: str | None = None


class TextRunRequest(BaseModel):
    event: dict[str, Any] | None = None
    event_text: str | None = None
    options: RunOptions = Field(default_factory=RunOptions)


class EvalRunRequest(BaseModel):
    options: RunOptions = Field(default_factory=lambda: RunOptions(event_limit=200))


class RunStatus(BaseModel):
    run_id: str
    mode: Literal["text", "image", "eval"]
    state: Literal["queued", "running", "completed", "failed"]
    created_at: str
    updated_at: str
    total: int
    completed: int = 0
    failed: int = 0
    current_event_id: str | None = None
    output_dir: str
    trace_dir: str
    results: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)


def create_app() -> FastAPI:
    app = FastAPI(title="Influence Strategy Web Console", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://127.0.0.1:8000",
            "http://localhost:8000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "workspace_root": str(PROJECT_ROOT),
            "default_events_path": str(DEFAULT_EVENTS_PATH),
            "web_runs_root": str(WEB_RUNS_ROOT),
            "llm_default_enabled": True,
        }

    @app.get("/api/event-template")
    def event_template() -> dict[str, Any]:
        return {
            "event_id": "web_event_001",
            "domain": "general",
            "event_title": "请输入事件标题",
            "event_summary": "请输入事件概要",
            "target": "引导公众理解事件影响并进行理性讨论。",
            "is_synthetic": False,
            "opinion_variants": [
                "请输入可用于发帖内容生成的观点变体。",
            ],
        }

    @app.post("/api/runs/text", response_model=RunStatus)
    def run_text_event(request: TextRunRequest) -> dict[str, Any]:
        event = _event_from_text_request(request)
        status = _create_run(mode="text", total=1)
        _submit_job(
            status["run_id"],
            lambda: _run_events_job(
                run_id=status["run_id"],
                events=[event],
                options=request.options,
            ),
        )
        return _read_status(status["run_id"])

    @app.post("/api/runs/image", response_model=RunStatus)
    def run_image_event(
        image: UploadFile = File(...),
        options: str = Form("{}"),
    ) -> dict[str, Any]:
        parsed_options = _parse_options(options)
        suffix = Path(image.filename or "").suffix.lower()
        if suffix not in SUPPORTED_IMAGE_SUFFIXES:
            raise HTTPException(status_code=400, detail=f"Unsupported image type: {suffix or image.filename}")

        status = _create_run(mode="image", total=1)
        run_dir = _run_dir(status["run_id"])
        upload_dir = run_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        image_path = upload_dir / f"input{suffix}"
        with image_path.open("wb") as handle:
            shutil.copyfileobj(image.file, handle)

        _submit_job(
            status["run_id"],
            lambda: _run_image_job(
                run_id=status["run_id"],
                image_path=image_path,
                options=parsed_options,
            ),
        )
        return _read_status(status["run_id"])

    @app.post("/api/eval-runs", response_model=RunStatus)
    def run_eval_events(request: EvalRunRequest | None = None) -> dict[str, Any]:
        options = (request.options if request is not None else RunOptions(event_limit=200))
        if options.event_limit is None:
            options.event_limit = 200
        events = select_hot_events(
            load_hot_events(DEFAULT_EVENTS_PATH),
            event_id=options.event_id,
            event_limit=options.event_limit,
        )
        status = _create_run(mode="eval", total=len(events))
        _submit_job(
            status["run_id"],
            lambda: _run_events_job(
                run_id=status["run_id"],
                events=events,
                options=options,
            ),
        )
        return _read_status(status["run_id"])

    @app.get("/api/runs")
    def list_runs() -> dict[str, Any]:
        WEB_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        statuses = []
        for status_path in sorted(WEB_RUNS_ROOT.glob("*/status.json"), reverse=True):
            try:
                statuses.append(_load_json(status_path))
            except (OSError, json.JSONDecodeError):
                continue
        return {"runs": statuses[:50]}

    @app.get("/api/runs/{run_id}", response_model=RunStatus)
    def get_run_status(run_id: str) -> dict[str, Any]:
        return _read_status(run_id)

    @app.get("/api/runs/{run_id}/results")
    def get_run_results(run_id: str) -> dict[str, Any]:
        status = _read_status(run_id)
        return {"run_id": run_id, "results": status.get("results", [])}

    @app.get("/api/runs/{run_id}/results/{event_id}")
    def get_result_detail(run_id: str, event_id: str) -> dict[str, Any]:
        output_path = _run_dir(run_id) / "output" / f"{event_id}_strategy_output.json"
        if not output_path.exists():
            raise HTTPException(status_code=404, detail=f"Result not found: {event_id}")
        return _load_json(output_path)

    @app.get("/api/runs/{run_id}/traces/{event_id}")
    def get_trace_detail(run_id: str, event_id: str) -> dict[str, Any]:
        trace_dir = _run_dir(run_id) / "trace" / event_id
        if not trace_dir.exists():
            raise HTTPException(status_code=404, detail=f"Trace not found: {event_id}")
        files = []
        for path in sorted(trace_dir.glob("*.json")):
            try:
                payload = _load_json(path)
            except json.JSONDecodeError:
                payload = path.read_text(encoding="utf-8")
            files.append({"name": path.name, "payload": payload})
        return {"run_id": run_id, "event_id": event_id, "files": files}

    @app.get("/api/runs/{run_id}/files/{event_id}")
    def download_result(run_id: str, event_id: str) -> FileResponse:
        output_path = _run_dir(run_id) / "output" / f"{event_id}_strategy_output.json"
        if not output_path.exists():
            raise HTTPException(status_code=404, detail=f"Result not found: {event_id}")
        return FileResponse(output_path, filename=output_path.name, media_type="application/json")

    if FRONTEND_DIST.exists():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")

    return app


app = create_app()


def _event_from_text_request(request: TextRunRequest) -> dict[str, Any]:
    if request.event is not None and request.event_text:
        raise HTTPException(status_code=400, detail="Use either event JSON or event text, not both.")
    if request.event is None and not request.event_text:
        raise HTTPException(status_code=400, detail="Event input is required.")
    if request.event is not None:
        return _normalize_hot_event(dict(request.event))
    text = str(request.event_text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Event text is empty.")
    return _normalize_hot_event(
        {
            "event_id": _safe_event_id(f"web_event_{_timestamp_id()}"),
            "domain": "general",
            "event_title": text[:48] or "新事件",
            "event_summary": text,
            "target": "引导公众理解事件影响并进行理性讨论。",
            "is_synthetic": False,
            "opinion_variants": [text],
        }
    )


def _normalize_hot_event(event: dict[str, Any]) -> dict[str, Any]:
    event_id = _safe_event_id(str(event.get("event_id") or f"web_event_{_timestamp_id()}"))
    title = str(event.get("event_title") or event.get("event_summary") or event_id).strip()
    summary = str(event.get("event_summary") or title).strip()
    variants = event.get("opinion_variants") or [summary or title]
    if isinstance(variants, str):
        variants = [item.strip() for item in variants.splitlines() if item.strip()]
    if not isinstance(variants, list) or not variants:
        variants = [summary or title]
    normalized = dict(event)
    normalized.update(
        {
            "event_id": event_id,
            "domain": str(event.get("domain") or "general").strip() or "general",
            "event_title": title,
            "event_summary": summary,
            "target": str(event.get("target") or "引导公众理解事件影响并进行理性讨论。").strip(),
            "is_synthetic": bool(event.get("is_synthetic", False)),
            "opinion_variants": [str(item).strip() for item in variants if str(item).strip()],
        }
    )
    if not normalized["event_title"] and not normalized["event_summary"]:
        raise HTTPException(status_code=400, detail="event_title or event_summary is required.")
    return normalized


def _parse_options(raw_options: str) -> RunOptions:
    try:
        payload = json.loads(raw_options or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid options JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Options must be a JSON object.")
    return RunOptions.model_validate(payload)


def _create_run(*, mode: Literal["text", "image", "eval"], total: int) -> dict[str, Any]:
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_dir = _run_dir(run_id)
    output_dir = run_dir / "output"
    trace_dir = run_dir / "trace"
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    now = _now()
    status = RunStatus(
        run_id=run_id,
        mode=mode,
        state="queued",
        created_at=now,
        updated_at=now,
        total=total,
        output_dir=str(output_dir),
        trace_dir=str(trace_dir),
    ).model_dump(mode="json")
    _write_status(run_id, status)
    return status


def _submit_job(run_id: str, job: Any) -> None:
    _EXECUTOR.submit(_run_job_wrapper, run_id, job)


def _run_job_wrapper(run_id: str, job: Any) -> None:
    try:
        _update_status(run_id, state="running")
        job()
        status = _read_status(run_id)
        final_state = "failed" if status.get("completed", 0) == 0 and status.get("failed", 0) else "completed"
        _update_status(run_id, state=final_state, current_event_id=None)
    except Exception as exc:  # pragma: no cover - defensive boundary for background jobs
        _append_error(run_id, event_id=None, message=f"{type(exc).__name__}: {exc}")
        _update_status(run_id, state="failed", current_event_id=None)


def _run_image_job(*, run_id: str, image_path: Path, options: RunOptions) -> None:
    _update_status(run_id, current_event_id="image_recognition")
    reference_events = load_hot_events(DEFAULT_EVENTS_PATH)
    events = load_image_events(
        image=image_path,
        image_dir=None,
        reference_events=reference_events,
        workspace_root=PROJECT_ROOT,
        event_limit=1,
        event_id=options.event_id,
        use_llm=options.use_llm,
    )
    _update_status(run_id, total=len(events))
    _run_events_job(run_id=run_id, events=events, options=options)


def _run_events_job(*, run_id: str, events: list[dict[str, Any]], options: RunOptions) -> None:
    output_dir = _run_dir(run_id) / "output"
    trace_dir = _run_dir(run_id) / "trace"
    for event in events:
        normalized_event = _normalize_hot_event(event)
        event_id = str(normalized_event["event_id"])
        _update_status(run_id, current_event_id=event_id)
        try:
            output_path, payload = _run_hot_event_evaluation(
                workspace_root=PROJECT_ROOT,
                output_dir=output_dir,
                hot_event=normalized_event,
                profile_limit=options.profile_limit,
                max_selected_nodes=options.max_selected_nodes,
                risk_level=options.risk_level,
                campaign_window_hours=options.campaign_window_hours,
                max_frequency_per_day=options.max_frequency_per_day,
                allowed_platforms=options.allowed_platforms,
                use_llm=options.use_llm,
                trace_dir=trace_dir,
            )
            _append_result(run_id, _result_summary(output_path=output_path, payload=payload))
        except Exception as exc:
            _append_error(run_id, event_id=event_id, message=f"{type(exc).__name__}: {exc}")


def _result_summary(*, output_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    event_id = output_path.name.replace("_strategy_output.json", "")
    strategy = payload.get("五维调度策略", {})
    target_object = strategy.get("目标对象", {}) if isinstance(strategy, dict) else {}
    content = strategy.get("内容", {}) if isinstance(strategy, dict) else {}
    selected_ids = target_object.get("选取数字人id组", []) if isinstance(target_object, dict) else []
    return {
        "event_id": event_id,
        "event_name": payload.get("事件名称", event_id),
        "selected_digital_human_ids": selected_ids,
        "content_llm": content.get("内容生成诊断", {}) if isinstance(content, dict) else {},
        "json_output_path": str(output_path),
    }


def _append_result(run_id: str, result: dict[str, Any]) -> None:
    with _LOCK:
        status = _read_status_unlocked(run_id)
        status.setdefault("results", []).append(result)
        status["completed"] = int(status.get("completed", 0)) + 1
        status["updated_at"] = _now()
        _write_status_unlocked(run_id, status)


def _append_error(run_id: str, *, event_id: str | None, message: str) -> None:
    with _LOCK:
        status = _read_status_unlocked(run_id)
        status.setdefault("errors", []).append({"event_id": event_id, "message": message})
        status["failed"] = int(status.get("failed", 0)) + 1
        status["updated_at"] = _now()
        _write_status_unlocked(run_id, status)


def _update_status(run_id: str, **updates: Any) -> None:
    with _LOCK:
        status = _read_status_unlocked(run_id)
        status.update(updates)
        status["updated_at"] = _now()
        _write_status_unlocked(run_id, status)


def _read_status(run_id: str) -> dict[str, Any]:
    with _LOCK:
        return _read_status_unlocked(run_id)


def _read_status_unlocked(run_id: str) -> dict[str, Any]:
    if run_id in _RUNS:
        return dict(_RUNS[run_id])
    status_path = _run_dir(run_id) / "status.json"
    if not status_path.exists():
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    status = _load_json(status_path)
    _RUNS[run_id] = status
    return dict(status)


def _write_status(run_id: str, status: dict[str, Any]) -> None:
    with _LOCK:
        _write_status_unlocked(run_id, status)


def _write_status_unlocked(run_id: str, status: dict[str, Any]) -> None:
    _RUNS[run_id] = dict(status)
    status_path = _run_dir(run_id) / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_dir(run_id: str) -> Path:
    return WEB_RUNS_ROOT / _safe_run_id(run_id)


def _safe_run_id(run_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", run_id)


def _safe_event_id(event_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", event_id.strip())
    return cleaned or f"web_event_{_timestamp_id()}"


def _timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

