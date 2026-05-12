import json
import os
import queue
import re
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

from .settings import Settings


def _open_worker_log_handle(log_path: str = "") -> Optional[TextIO]:
    log_path = (log_path or "").strip()
    if not log_path:
        return None
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "a", encoding="utf-8", buffering=1)


def _worker_log(log_handle: Optional[TextIO], message: str, **payload: Any) -> None:
    if not log_handle:
        return
    stamp = datetime.now().isoformat(timespec="seconds")
    if payload:
        log_handle.write(f"[{stamp}] {message} | {json.dumps(payload, ensure_ascii=False)}\n")
    else:
        log_handle.write(f"[{stamp}] {message}\n")
    log_handle.flush()


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[。！？.!?])\s*|\n+", text)
    return [part.strip() for part in parts if part.strip()]


def recursive_chunk_with_overlap(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    paragraphs = [p for p in re.split(r"(?<=\n\n)", text) if p.strip()]
    chunks: List[str] = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= chunk_size:
            current_chunk += paragraph + "\n\n"
            continue

        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + paragraph + "\n\n"
            continue

        for sentence in split_sentences(paragraph):
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence
                continue
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + sentence
            else:
                chunks.append(sentence)
                current_chunk = sentence[-overlap:] if len(sentence) > overlap else sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _summarize_parent_text(parent_text: str, limit: int = 260) -> str:
    sentences = split_sentences(parent_text)
    if not sentences:
        return _compact_whitespace(parent_text)[:limit]
    summary = " ".join(sentences[:2])
    return _compact_whitespace(summary)[:limit]


def _build_contextual_sentence(parent: Dict[str, Any], sentence: str) -> str:
    prefix_parts = [
        f"title: {parent.get('title', '')}",
        f"page_type: {parent.get('page_type', 'article')}",
        f"source_type: {parent.get('source_type', 'analysis')}",
    ]
    domain = str(parent.get("domain") or "").strip()
    if domain:
        prefix_parts.append(f"domain: {domain}")
    summary = str(parent.get("summary") or "").strip()
    if summary:
        prefix_parts.append(f"parent_summary: {summary}")
    prefix = " | ".join(part for part in prefix_parts if part.strip())
    return f"{prefix} | sentence: {_compact_whitespace(sentence)}"


def _build_parent_rerank_text(parent: Dict[str, Any]) -> str:
    parts = [
        f"title: {parent.get('title', '')}",
        f"page_type: {parent.get('page_type', 'article')}",
        f"source_type: {parent.get('source_type', 'analysis')}",
        f"summary: {parent.get('summary', '')}",
        f"content: {parent.get('text', '')}",
    ]
    return "\n".join(part for part in parts if str(part).strip())


def model_worker_process(
    settings_dict: Dict[str, Any],
    job_queue: Any,
    result_queue: Any,
    worker_state: Any,
    bootstrap_log_path: str = "",
) -> None:
    log_handle = _open_worker_log_handle(bootstrap_log_path)
    _worker_log(log_handle, "Worker process bootstrapping.", pid=os.getpid())

    settings = Settings(**settings_dict)
    _worker_log(
        log_handle,
        "Worker settings loaded.",
        report_dir=settings.report_dir,
        embedder_path=settings.embedder_path,
        reranker_path=settings.reranker_path,
    )
    heartbeat_stop = threading.Event()

    def publish_status(status: str, job_id: str = "", error: str = "") -> None:
        worker_state["status"] = status
        worker_state["job_id"] = job_id
        worker_state["error"] = error
        worker_state["heartbeat"] = time.time()
        _worker_log(log_handle, "Worker status changed.", status=status, job_id=job_id, error=error)

    def heartbeat_loop() -> None:
        while not heartbeat_stop.is_set():
            try:
                worker_state["heartbeat"] = time.time()
            except Exception:
                pass
            heartbeat_stop.wait(settings.worker_heartbeat_interval)

    try:
        publish_status("IMPORTING_TRANSFORMERS")
        import transformers

        publish_status("IMPORTING_TORCH")
        import torch

        publish_status("IMPORTING_SENTENCE_TRANSFORMERS")
        from sentence_transformers import CrossEncoder, SentenceTransformer

        publish_status("IMPORTING_FAISS")
        import faiss

        publish_status("CONFIGURING_TRANSFORMERS")
        transformers.utils.logging.set_verbosity_error()

        publish_status("CHECKING_DEVICE")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            publish_status("WARMING_UP_CUDA")
            _ = torch.zeros(1).cuda()

        publish_status("LOADING_EMBEDDER")
        embedder = SentenceTransformer(settings.embedder_path, device=device, local_files_only=True)

        publish_status("LOADING_RERANKER")
        reranker = CrossEncoder(settings.reranker_path, device=device, local_files_only=True)

        publish_status("READY")
    except Exception as exc:
        heartbeat_stop.set()
        _worker_log(log_handle, "Worker bootstrap failed.", error=str(exc), traceback=traceback.format_exc())
        publish_status(f"ERROR: {exc}", error=str(exc))
        return

    heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    heartbeat_thread.start()

    while True:
        job_id = None
        try:
            task = job_queue.get(timeout=settings.worker_heartbeat_interval)
            if not task:
                continue

            job_id = str(task.get("job_id") or "")
            publish_status("BUSY", job_id=job_id)
            query_preview = str(task["queries"][0])[:240] if task.get("queries") else ""
            _worker_log(
                log_handle,
                "Worker accepted job.",
                job_id=job_id,
                query_count=len(task.get("queries", [])),
                document_count=len(task.get("documents", [])),
                query_preview=query_preview,
            )
            queries = task["queries"]
            documents = task["documents"]

            parent_chunks: List[Dict[str, Any]] = []
            child_chunks: List[Dict[str, Any]] = []
            parent_id = 0

            for document in documents:
                content = str(document.get("content") or "").strip()
                if not content:
                    continue
                for parent_text in recursive_chunk_with_overlap(content):
                    summary = _summarize_parent_text(parent_text)
                    parent_record = {
                        "text": parent_text,
                        "summary": summary,
                        "url": document.get("url", "Unknown"),
                        "title": document.get("title", ""),
                        "published_time": document.get("published_time", ""),
                        "page_type": document.get("page_type", "article"),
                        "source_type": document.get("source_type", "analysis"),
                        "content_quality_score": float(document.get("content_quality_score", 0.0) or 0.0),
                        "domain": re.sub(r"^www\.", "", re.sub(r"^https?://", "", str(document.get("url") or "")).split("/", 1)[0].lower()),
                    }
                    parent_chunks.append(parent_record)
                    for sentence in split_sentences(parent_text):
                        if len(sentence) < 15:
                            continue
                        child_chunks.append(
                            {
                                "embedding_text": _build_contextual_sentence(parent_record, sentence),
                                "sentence": sentence,
                                "parent_id": parent_id,
                            }
                        )
                    parent_id += 1

            if not child_chunks:
                result_queue.put({"job_id": job_id, "status": "EMPTY", "evidence": []})
                _worker_log(log_handle, "Worker produced empty result.", job_id=job_id)
                publish_status("READY")
                continue

            child_embeddings = embedder.encode(
                [item["embedding_text"] for item in child_chunks],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            index = faiss.IndexFlatIP(child_embeddings.shape[1])
            index.add(child_embeddings)
            query_embeddings = embedder.encode(queries, normalize_embeddings=True, show_progress_bar=False)
            _, indices = index.search(query_embeddings, min(settings.search_max_results * 8, len(child_chunks)))

            recall_parent_indices = set()
            for row in indices:
                for idx in row:
                    if idx != -1:
                        recall_parent_indices.add(int(child_chunks[idx]["parent_id"]))

            recall_parents = [parent_chunks[idx] for idx in sorted(recall_parent_indices)]
            if not recall_parents:
                result_queue.put({"job_id": job_id, "status": "EMPTY", "evidence": []})
                publish_status("READY")
                continue

            pair_items = []
            for current_query in queries:
                for parent in recall_parents:
                    pair_items.append({"pair": [current_query, _build_parent_rerank_text(parent)], "parent": parent})

            scores = reranker.predict([item["pair"] for item in pair_items], show_progress_bar=False)
            scored = []
            for score, item in zip(scores, pair_items):
                parent = item["parent"]
                scored.append(
                    {
                        "score": float(score),
                        "url": parent["url"],
                        "title": parent.get("title", ""),
                        "published_time": parent.get("published_time", ""),
                        "page_type": parent.get("page_type", "article"),
                        "source_type": parent.get("source_type", "analysis"),
                        "content_quality_score": float(parent.get("content_quality_score", 0.0) or 0.0),
                        "excerpt": parent["text"],
                    }
                )

            scored.sort(key=lambda item: item["score"], reverse=True)
            evidence = []
            seen = set()
            for item in scored:
                dedupe_key = (item["url"], item["excerpt"])
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                evidence.append(item)
                if len(evidence) >= settings.search_max_results + 1:
                    break

            result_queue.put({"job_id": job_id, "status": "OK", "evidence": evidence})
            _worker_log(log_handle, "Worker finished job.", job_id=job_id, evidence_count=len(evidence))
        except queue.Empty:
            continue
        except Exception as exc:
            if job_id:
                _worker_log(
                    log_handle,
                    "Worker job failed.",
                    job_id=job_id,
                    error=str(exc),
                    traceback=traceback.format_exc(),
                )
                result_queue.put(
                    {
                        "job_id": job_id,
                        "status": "ERROR",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
        finally:
            publish_status("READY")
