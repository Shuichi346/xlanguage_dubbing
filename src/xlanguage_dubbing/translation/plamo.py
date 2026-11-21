"""Local PLaMo 3 Translate 8B Q4_K_M inference and resumable translation."""

from __future__ import annotations

import atexit
import hashlib
import json
import re
import secrets
import shutil
import socket
import subprocess
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

from xlanguage_dubbing import config
from xlanguage_dubbing.audio.segment_io import load_segments_json, save_segments_json_atomic
from xlanguage_dubbing.config import (
    INPUT_REPEAT_THRESHOLD,
    OUTPUT_LANG,
    OUTPUT_REPEAT_THRESHOLD,
    PLAMO_TRANSLATE_FILE,
    PLAMO_TRANSLATE_N_CTX,
    PLAMO_TRANSLATE_N_GPU_LAYERS,
    PLAMO_TRANSLATE_REPEAT_PENALTY,
    PLAMO_TRANSLATE_REPO,
    PLAMO_TRANSLATE_RETRIES,
    PLAMO_TRANSLATE_RETRY_BACKOFF_SEC,
    PLAMO_TRANSLATE_SERVER_BIN,
)
from xlanguage_dubbing.core.models import Segment
from xlanguage_dubbing.core.progress import ProgressStore
from xlanguage_dubbing.lang_utils import (
    detect_language_from_text,
    normalize_lang_code,
    validate_translation_language,
    validate_translation_languages,
)
from xlanguage_dubbing.utils import (
    PipelineError,
    atomic_write_json,
    force_memory_cleanup,
    load_json_if_exists,
    print_step,
)


class PlamoTranslateError(Exception):
    """翻訳エラー。"""


# 翻訳失敗時のプレースホルダーテキスト
_TRANSLATION_ERROR_MARKER = "翻訳エラー"
_REPETITIVE_INPUT_MARKER = "（繰り返し音声）"
_REPETITIVE_PART_MARKER = "繰り返し音声エラー"

_SERVER = None


def translation_namespace() -> str:
    """Isolate PLaMo checkpoints and outputs from old models and other settings."""
    settings = {
        "model": config.PLAMO_TRANSLATE_REPO,
        "file": config.PLAMO_TRANSLATE_FILE,
        "prompt": "pfml3-v1",
        "source": normalize_lang_code(config.INPUT_LANG),
        "target": normalize_lang_code(config.OUTPUT_LANG),
        "context": config.PLAMO_TRANSLATE_N_CTX,
        "repeat_penalty": config.PLAMO_TRANSLATE_REPEAT_PENALTY,
    }
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    return f"plamo3_{settings['target']}_{fingerprint}"


def _get_model_path() -> str:
    """Download or reuse the exact official Q4_K_M model under project models/."""
    from huggingface_hub import hf_hub_download

    print_step(f"  モデルを取得中: {PLAMO_TRANSLATE_REPO}/{PLAMO_TRANSLATE_FILE}")
    model_dir = Path(__file__).resolve().parents[3] / "models" / "plamo-3-translate-8b-GGUF"
    return hf_hub_download(
        repo_id=PLAMO_TRANSLATE_REPO, filename=PLAMO_TRANSLATE_FILE, local_dir=model_dir,
    )


def resolve_llama_server() -> str:
    """Resolve the installed llama.cpp server, including the Homebrew prefix."""
    binary = str(Path(PLAMO_TRANSLATE_SERVER_BIN).expanduser())
    resolved = shutil.which(binary)
    if resolved:
        return resolved
    if binary == "llama-server" and Path("/opt/homebrew/bin/llama-server").is_file():
        return "/opt/homebrew/bin/llama-server"
    raise PipelineError("llama-server が見つかりません。brew install llama.cpp を実行してください。")


class _PlamoServer:
    """Own exactly one private localhost llama-server process for translation."""

    def __init__(self) -> None:
        if not 256 <= PLAMO_TRANSLATE_N_CTX <= 32768:
            raise PipelineError("PLAMO_TRANSLATE_N_CTX は 256〜32768 で指定してください。")
        binary = resolve_llama_server()
        model_path = _get_model_path()
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"
        self.api_key = secrets.token_urlsafe(32)
        self.opener = build_opener(ProxyHandler({}))
        self.log = tempfile.TemporaryFile()
        command = [
            binary, "--model", model_path, "--ctx-size", str(PLAMO_TRANSLATE_N_CTX),
            "--gpu-layers", str(PLAMO_TRANSLATE_N_GPU_LAYERS),
            "--host", "127.0.0.1", "--port", str(port), "--parallel", "1",
            "--no-webui", "--api-key", self.api_key,
        ]
        print_step("  PLaMo 3 Translate 8B の llama-server を起動中...")
        try:
            self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT)
        except BaseException:
            self.log.close()
            raise
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    self.log.seek(0)
                    detail = self.log.read().decode("utf-8", errors="replace")[-4000:]
                    raise PipelineError(f"PLaMo llama-server の起動に失敗しました。\n{detail}")
                try:
                    self.request("/health", timeout=2)
                    return
                except (URLError, TimeoutError):
                    time.sleep(0.25)
            raise PipelineError("PLaMo llama-server の起動がタイムアウトしました。")
        except BaseException:
            self.close()
            raise

    def request(self, path: str, payload: dict | None = None, *, timeout: float = 300) -> dict:
        request = Request(
            self.base_url + path,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with self.opener.open(request, timeout=timeout) as response:
            return json.load(response)

    def close(self) -> None:
        """Stop only this process; never discover or stop an external server."""
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
        finally:
            self.log.close()


def _get_server() -> _PlamoServer:
    global _SERVER
    if _SERVER is None:
        _SERVER = _PlamoServer()
    return _SERVER


def release_translation_model() -> None:
    """Stop the owned translation server before TTS to release CPU/Metal memory."""
    global _SERVER
    if _SERVER is not None:
        _SERVER.close()
        _SERVER = None
        force_memory_cleanup()
        print_step("  PLaMo 3 Translate 8B サーバーを終了しました")


atexit.register(release_translation_model)


# =========================================================
# 翻訳異常検出
# =========================================================


def _has_consecutive_repeats(phrases: list[str], threshold: int) -> bool:
    """フレーズリスト内に連続で threshold 回以上同じフレーズがあるか検査する。

    LLM が暴走して「the the the the」「ありますありますあります」のように
    同じフレーズを連続リピートするケースのみを検出する。
    英語の冠詞や前置詞など、文中に自然に複数回出現する単語は
    連続でなければ正常とみなす。
    """
    if len(phrases) < threshold:
        return False
    consecutive = 1
    for i in range(1, len(phrases)):
        if phrases[i] == phrases[i - 1]:
            consecutive += 1
            if consecutive >= threshold:
                return True
        else:
            consecutive = 1
    return False


def _detect_repeated_phrases(text: str) -> bool:
    """翻訳結果に同一フレーズが連続でN回以上繰り返されているか検査する。

    「the the the」「ありますありますあります」のような LLM の暴走パターンのみ検出する。
    文中の非連続な同一単語出現（英語の冠詞など自然な繰り返し）は検出しない。
    """
    t = (text or "").strip()
    if not t:
        return False

    phrases = re.split(r"[。、！？!?,.\s]+", t)
    phrases = [p.strip() for p in phrases if len(p.strip()) >= 2]

    if len(phrases) < OUTPUT_REPEAT_THRESHOLD:
        return False

    return _has_consecutive_repeats(phrases, OUTPUT_REPEAT_THRESHOLD)


def is_translation_glitch(text: str) -> bool:
    """翻訳結果が異常かどうかを判定する。"""
    return _detect_repeated_phrases(text)


def _is_translation_error_placeholder(text: str) -> bool:
    """テキストが翻訳エラーのプレースホルダーかどうかを判定する。"""
    t = (text or "").strip()
    return t in (
        _TRANSLATION_ERROR_MARKER,
        _REPETITIVE_INPUT_MARKER,
        _REPETITIVE_PART_MARKER,
    )


def _is_repetitive_input(text: str) -> bool:
    """入力テキストが連続繰り返しで翻訳不要かを判定する。

    LLM の暴走による連続リピートのみを検出する。
    文中に自然に同じ単語が複数回出現するケースは正常とみなす。
    """
    t = (text or "").strip()
    if not t:
        return True

    phrases = re.findall(r'"([^"]+)"', t)
    if not phrases:
        phrases = re.split(r"(?<=[\.\!\?])\s+", t)
        phrases = [p.strip() for p in phrases if p.strip()]

    if len(phrases) <= 2:
        return False

    # 連続出現のみ検出する
    phrases_lower = [p.strip().lower() for p in phrases]
    return _has_consecutive_repeats(phrases_lower, INPUT_REPEAT_THRESHOLD)


def _build_prompt(text: str, source_lang: str, target_lang: str) -> str:
    """Build the official PFML3 format without a textual BOS or trailing newline."""
    source, target = validate_translation_languages(source_lang, target_lang)
    return (
        "<|plamo:tag|>dataset<|plamo:msg|>translation"
        f"<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>{source}"
        f"<|plamo:msg|>{text.strip()}"
        f"<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>{target}<|plamo:msg|>"
    )


def _translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """Translate every supported pair through PLaMo, using greedy decoding."""
    prompt = _build_prompt(text, source_lang, target_lang)
    # llama.cpp parses PFML3 special tokens and adds the model-defined BOS.
    response = _get_server().request("/v1/completions", {
        "prompt": prompt,
        "max_tokens": PLAMO_TRANSLATE_N_CTX // 2,
        "temperature": 0.0,
        "top_p": 1.0,
        "repeat_penalty": PLAMO_TRANSLATE_REPEAT_PENALTY,
        "stop": ["<|plamo:tag|>", "<|plamo:bos|>"],
        "echo": False,
    })
    choices = response.get("choices", [])
    if not choices or choices[0].get("finish_reason") == "length":
        raise PlamoTranslateError("翻訳が完了しませんでした（出力上限）。")
    result = choices[0].get("text", "").strip()
    if not result:
        raise PlamoTranslateError("翻訳結果が空です。")
    return result


# =========================================================
# 翻訳クライアント
# =========================================================


class PlamoTranslateClient:
    """Translate supported language pairs with PLaMo 3 Translate 8B only."""

    def translate(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "",
        retries: int = PLAMO_TRANSLATE_RETRIES,
        retry_backoff_sec: float = PLAMO_TRANSLATE_RETRY_BACKOFF_SEC,
    ) -> tuple[str, str]:
        """テキストを翻訳する。"""
        source_lang, tgt = validate_translation_languages(source_lang, target_lang or OUTPUT_LANG)
        if retries < 1:
            raise ValueError("retries must be at least 1")

        last_err: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                result = _translate_text(text, source_lang, tgt)
                return result, "stop"
            except Exception as exc:
                last_err = exc
                if attempt < retries:
                    wait_sec = retry_backoff_sec * attempt
                    print_step(
                        f"    翻訳リトライ {attempt}/{retries}: "
                        f"{wait_sec:.1f}秒待機... ({exc})"
                    )
                    time.sleep(wait_sec)

        raise PlamoTranslateError(f"翻訳失敗(リトライ枯渇): {last_err}")


# =========================================================
# セグメント翻訳
# =========================================================


def _new_segment_src_only(seg: Segment) -> Segment:
    """元テキストのみを保持したセグメントを生成する。"""
    return replace(seg, text_tgt="")


def translate_segment_safely(
    client: PlamoTranslateClient,
    text_src: str,
    source_lang: str,
    target_lang: str,
) -> str:
    """セグメントを安全に翻訳する。"""
    source_lang, target_lang = validate_translation_languages(source_lang, target_lang)
    t = (text_src or "").strip()
    if not t:
        return ""

    if _is_repetitive_input(t):
        print_step("    繰り返し入力を検出 → スキップ")
        return _REPETITIVE_INPUT_MARKER

    if len(t) <= 2000:
        out, _ = client.translate(
            t, source_lang=source_lang, target_lang=target_lang
        )
        tgt = out.strip()
        if is_translation_glitch(tgt):
            return _TRANSLATION_ERROR_MARKER
        return tgt

    parts = re.split(r"(?<=[\.\!\?])\s+", t)
    outs: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if _is_repetitive_input(p):
            outs.append(_REPETITIVE_PART_MARKER)
            continue
        out, _ = client.translate(
            p, source_lang=source_lang, target_lang=target_lang
        )
        tgt = out.strip()
        if is_translation_glitch(tgt):
            outs.append(_TRANSLATION_ERROR_MARKER)
        else:
            outs.append(tgt)

    joined = " ".join([o for o in outs if o])
    if is_translation_glitch(joined):
        return _TRANSLATION_ERROR_MARKER
    return joined


def translate_segments_resumable(
    client: PlamoTranslateClient,
    segments_src: list[Segment],
    seg_json_path: Path,
    progress: ProgressStore,
    detected_lang: str = "",
) -> list[Segment]:
    """セグメントを再開可能な形式で翻訳する。

    各セグメントの detected_lang を翻訳の source_lang として使用する。
    detected_lang が空の場合はセグメントのテキストから自動検出する。

    以前の実行で「翻訳エラー」となったセグメントは再翻訳を試みる。
    """
    target_lang = validate_translation_language(OUTPUT_LANG)
    for seg in segments_src:
        validate_translation_language(
            seg.detected_lang or detected_lang or detect_language_from_text(seg.text_src) or "en"
        )

    namespace = translation_namespace()
    can_resume = progress.data.get("artifacts", {}).get("translation_namespace") == namespace
    if can_resume and seg_json_path.exists():
        try:
            loaded = load_segments_json(seg_json_path)
            if [replace(s, text_tgt="") for s in loaded] == [replace(s, text_tgt="") for s in segments_src]:
                segments = loaded
            else:
                segments = [_new_segment_src_only(s) for s in segments_src]
        except Exception:
            segments = [_new_segment_src_only(s) for s in segments_src]
    else:
        segments = [_new_segment_src_only(s) for s in segments_src]

    record_dir = seg_json_path.with_suffix(".records")
    if can_resume:
        for segno, source in enumerate(segments_src, start=1):
            record = load_json_if_exists(record_dir / f"{segno:05d}.json")
            if isinstance(record, dict) and record.get("source") == asdict(_new_segment_src_only(source)):
                text = record.get("translation")
                if isinstance(text, str):
                    segments[segno - 1] = replace(source, text_tgt=text)

    # Publish cleared source records before marking this checkpoint as PLaMo-owned.
    # A crash before the first inference must not legitimize a previous model's text.
    save_segments_json_atomic(segments, seg_json_path)
    progress.set_artifact("translation_namespace", namespace)
    progress.set_step("translate_done", False)
    progress.save()
    total = len(segments)

    for segno, seg in enumerate(segments, start=1):
        text_tgt = (seg.text_tgt or "").strip()

        # 翻訳済みかつ正常な結果はスキップする。
        # ただし「翻訳エラー」プレースホルダーの場合は再翻訳を試みる。
        if text_tgt and not _is_translation_error_placeholder(text_tgt):
            if is_translation_glitch(text_tgt):
                # glitch 検出されたが定型プレースホルダーではない → 再翻訳対象
                pass
            else:
                continue

        # セグメント単位の元言語を決定する
        seg_source_lang = (
            seg.detected_lang
            or detected_lang
            or detect_language_from_text(seg.text_src)
            or "en"
        )

        retry_label = "（再翻訳）" if text_tgt else ""
        print_step(
            f"  翻訳 {segno}/{total}: {seg.start:.3f}-{seg.end:.3f} "
            f"[{seg_source_lang}→{target_lang}] "
            f"'{seg.text_src[:60]}'{retry_label}"
        )
        tgt = translate_segment_safely(
            client, seg.text_src, seg_source_lang, target_lang
        )
        segments[segno - 1] = replace(seg, text_tgt=tgt)
        atomic_write_json(record_dir / f"{segno:05d}.json", {
            "source": asdict(_new_segment_src_only(seg)), "translation": tgt,
        })
        progress.set_step("translate_done", False)
        progress.set_artifact("segments_translated_json", str(seg_json_path))
        progress.save()

    progress.set_step("translate_done", True)
    save_segments_json_atomic(segments, seg_json_path)
    progress.save()
    return segments
