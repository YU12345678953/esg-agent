import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class SessionStore:
    """Single write path for UI-facing session state and messages."""

    def __init__(self, sessions_dir: Path) -> None:
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def session_dir(self, session_id: str) -> Path:
        return self.sessions_dir / session_id

    def session_json_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "session.json"

    def messages_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "messages.json"

    def load(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            if session_id in self._cache:
                return self._cache[session_id]

            path = self.session_json_path(session_id)
            if not path.exists():
                raise FileNotFoundError(f"Session not found: {session_id}")

            self._cache[session_id] = json.loads(path.read_text(encoding="utf-8"))
            return self._cache[session_id]

    def create(self, session_id: str, session: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            self._cache[session_id] = dict(session)
            self.save(session_id)
            return self._cache[session_id]

    def save(self, session_id: str, session: Dict[str, Any] | None = None) -> None:
        with self._lock:
            if session is not None:
                self._cache[session_id] = session
            if session_id not in self._cache:
                raise FileNotFoundError(f"Session not loaded: {session_id}")

            path = self.session_json_path(session_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(".json.tmp")
            tmp_path.write_text(
                json.dumps(self._cache[session_id], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp_path.replace(path)

    def update(self, session_id: str, **updates: Any) -> Dict[str, Any]:
        with self._lock:
            session = self.load(session_id)
            session.update(updates)
            session["updated_at"] = now_text()
            self.save(session_id)
            return session

    def mutate(self, session_id: str, mutator) -> Dict[str, Any]:
        with self._lock:
            session = self.load(session_id)
            mutator(session)
            session["updated_at"] = now_text()
            self.save(session_id)
            return session

    def append_message(self, session_id: str, role: str, content: str, **extra: Any) -> Dict[str, Any]:
        with self._lock:
            path = self.messages_path(session_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            messages = self.get_messages(session_id)
            message = {
                "role": role,
                "content": content,
                "created_at": now_text(),
                **extra,
            }
            messages.append(message)
            tmp_path = path.with_suffix(".json.tmp")
            tmp_path.write_text(
                json.dumps(messages, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp_path.replace(path)
            return message

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        path = self.messages_path(session_id)
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))
