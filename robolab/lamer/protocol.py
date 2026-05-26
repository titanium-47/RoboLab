"""
Wire protocol for the remote environment server.

Uses TCP sockets with length-prefixed pickle frames.
All data crossing this boundary (numpy arrays, dicts, lists, strings)
is natively pickle-serializable.

Wire format:
    [4 bytes: payload length (big-endian uint32)] [N bytes: pickle payload]

Copied from LaMer/agent_system/environments/remote/protocol.py (pure stdlib).
"""

import io
import pickle
import struct
import socket
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------

@dataclass
class EnvRequest:
    """Client -> Server request."""
    request_id: str
    method: str  # "reset" | "step" | "restart" | "reflect" |
                 # "success_evaluator" | "get_properties" | "close"
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)


@dataclass
class EnvResponse:
    """Server -> Client response."""
    request_id: str
    status: str  # "ok" | "error"
    result: Any = None
    error_message: str = ""


# ---------------------------------------------------------------------------
# Wire helpers
# ---------------------------------------------------------------------------

_HEADER_FMT = "!I"  # big-endian unsigned 32-bit int
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_MAX_MESSAGE_SIZE = 2 ** 32 - 1  # ~4 GB


def send_message(sock: socket.socket, obj: Any) -> None:
    """Serialize *obj* with pickle and send as a length-prefixed frame."""
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    length = len(payload)
    if length > _MAX_MESSAGE_SIZE:
        raise ValueError(
            f"Message too large: {length} bytes (max {_MAX_MESSAGE_SIZE})"
        )
    header = struct.pack(_HEADER_FMT, length)
    sock.sendall(header + payload)


class _CrossVenvUnpickler(pickle.Unpickler):
    """Unpickler that resolves EnvRequest/EnvResponse from any module path.

    The LaMer client pickles these from ``agent_system.environments.remote.protocol``
    but the language-table server doesn't have that module.  This maps both
    known module paths to the local classes so deserialization works across venvs.
    """

    _CLASS_MAP = {
        ("agent_system.environments.remote.protocol", "EnvRequest"): EnvRequest,
        ("agent_system.environments.remote.protocol", "EnvResponse"): EnvResponse,
    }

    def find_class(self, module: str, name: str):
        key = (module, name)
        if key in self._CLASS_MAP:
            return self._CLASS_MAP[key]
        return super().find_class(module, name)


def recv_message(sock: socket.socket) -> Any:
    """Receive a length-prefixed frame and deserialize with pickle."""
    header = _recv_exact(sock, _HEADER_SIZE)
    (length,) = struct.unpack(_HEADER_FMT, header)
    payload = _recv_exact(sock, length)
    return _CrossVenvUnpickler(io.BytesIO(payload)).load()


def _recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    """Read exactly *nbytes* from *sock*, raising on premature EOF."""
    chunks = []
    remaining = nbytes
    while remaining > 0:
        chunk = sock.recv(min(remaining, 4 * 1024 * 1024))  # 4 MB reads
        if not chunk:
            raise ConnectionError(
                f"Connection closed while reading "
                f"({nbytes - remaining}/{nbytes} bytes received)"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)