import os
import importlib
import threading
from typing import Any, Dict, Optional


class LangfuseMonitor:
    """Langfuse 轻量封装，确保未安装或未配置时不影响主链路。"""

    def __init__(self, service_name: str = "smart-agent"):
        self.service_name = service_name
        self.enabled = False
        self._client = None
        self._flush_lock = threading.Lock()
        self._flush_inflight = False
        self._flush_pending = False

        enabled_flag = os.getenv("LANGFUSE_ENABLED", "true").strip().lower()
        if enabled_flag in {"0", "false", "off", "no"}:
            return

        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
        timeout_raw = os.getenv("LANGFUSE_TIMEOUT_SECONDS", "8").strip()
        try:
            timeout_seconds = max(1, int(timeout_raw))
        except ValueError:
            timeout_seconds = 8

        if not public_key or not secret_key:
            return

        try:
            langfuse_module = importlib.import_module("langfuse")
            Langfuse = getattr(langfuse_module, "Langfuse")

            self._client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
                timeout=timeout_seconds,
            )
            self.enabled = True
            print(f"[Langfuse] 已启用, host={host}, timeout={timeout_seconds}s")
        except Exception as exc:
            print(f"[Langfuse] 初始化失败，已自动降级为无监控模式: {exc}")

    def start_trace(
        self,
        *,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        input_payload: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if not self.enabled or self._client is None:
            return None

        trace_meta: Dict[str, Any] = dict(metadata or {})

        kwargs: Dict[str, Any] = {
            "as_type": "span",
            "name": name,
        }
        if input_payload is not None:
            kwargs["input"] = input_payload
        if trace_meta:
            kwargs["metadata"] = trace_meta

        try:
            # 使用 start_observation 返回真实 observation 对象，避免 context manager 类型不匹配。
            observation = self._client.start_observation(**kwargs)

            # 将 user_id/session_id 写入 trace 顶层字段，确保 Langfuse 能按 session 聚合。
            trace_update: Dict[str, Any] = {}
            if user_id:
                trace_update["user_id"] = user_id
            if session_id:
                trace_update["session_id"] = session_id

            if trace_update:
                try:
                    update_trace = getattr(observation, "update_trace", None)
                    if callable(update_trace):
                        update_trace(**trace_update)
                    else:
                        fallback_update = getattr(self._client, "update_current_trace", None)
                        if callable(fallback_update):
                            fallback_update(**trace_update)
                except Exception as exc:
                    print(f"[Langfuse] 更新 trace 字段失败: {exc}")

            return observation
        except Exception as exc:
            print(f"[Langfuse] 创建 trace 失败: {exc}")
            return None

    def start_span(self, parent: Any, *, name: str, input_payload: Optional[Any] = None, metadata: Optional[Dict[str, Any]] = None):
        if parent is None or not self.enabled or self._client is None:
            return None

        kwargs: Dict[str, Any] = {
            "as_type": "span",
            "name": name
        }
        if input_payload is not None:
            kwargs["input"] = input_payload
        if metadata:
            kwargs["metadata"] = metadata

        try:
            return parent.start_observation(**kwargs)
        except Exception as exc:
            print(f"[Langfuse] 创建 span 失败: {exc}")
            return None

    def start_generation(
        self,
        parent: Any,
        *,
        name: str,
        model: str,
        input_payload: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if parent is None or not self.enabled or self._client is None:
            return None

        kwargs: Dict[str, Any] = {
            "as_type": "generation",
            "name": name,
            "model": model,
        }
        if input_payload is not None:
            kwargs["input"] = input_payload
        if metadata:
            kwargs["metadata"] = metadata

        try:
            return parent.start_observation(**kwargs)
        except Exception as exc:
            print(f"[Langfuse] 创建 generation 失败: {exc}")
            return None

    def update_observation(
        self,
        observation: Any,
        *,
        input_payload: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if observation is None:
            return

        kwargs: Dict[str, Any] = {}
        if input_payload is not None:
            kwargs["input"] = input_payload
        if output is not None:
            kwargs["output"] = output
        if metadata:
            kwargs["metadata"] = metadata

        method = getattr(observation, "update", None)
        if callable(method):
            try:
                method(**kwargs)
                return
            except TypeError:
                try:
                    method()
                    return
                except Exception:
                    return
            except Exception:
                return

    def end_observation(self, observation: Any, *, output: Optional[Any] = None, metadata: Optional[Dict[str, Any]] = None):
        if observation is None:
            return

        # Langfuse v3 的 end() 通常不接受 output/metadata，
        # 需要先 update，再调用 end() 结束 observation。
        if output is not None or metadata:
            self.update_observation(observation, output=output, metadata=metadata)

        end_method = getattr(observation, "end", None)
        if callable(end_method):
            try:
                end_method()
            except Exception:
                pass

    def _flush_sync(self):
        if not self.enabled or self._client is None:
            return

        try:
            self._client.flush()
        except Exception as exc:
            print(f"[Langfuse] flush 失败: {exc}")

    def _flush_worker(self):
        while True:
            self._flush_sync()
            with self._flush_lock:
                if self._flush_pending:
                    # flush 进行期间收到过新请求，继续追加一次 flush。
                    self._flush_pending = False
                    continue
                self._flush_inflight = False
                return

    def flush(self, background: bool = True):
        """默认后台异步 flush, 避免在主链路上因网络抖动产生阻塞。"""
        if not background:
            self._flush_sync()
            return

        if not self.enabled or self._client is None:
            return

        with self._flush_lock:
            if self._flush_inflight:
                self._flush_pending = True
                return
            self._flush_inflight = True
            self._flush_pending = False

        t = threading.Thread(target=self._flush_worker, daemon=True)
        t.start()

    def flush_sync(self):
        """强制同步 flush, 适合程序退出前确保发送完成。"""
        self.flush(background=False)