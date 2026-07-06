import json
import socket
from typing import Any, Dict

from agent.config import HEADER_SIZE


def encode_message(message: Dict[str, Any]) -> bytes:
    payload = json.dumps(message).encode("utf-8")
    header = len(payload).to_bytes(HEADER_SIZE, byteorder="big")
    return header + payload


def recv_exact(sock: socket.socket, size: int) -> bytes:
    data = b""

    while len(data) < size:
        packet = sock.recv(size - len(data))

        if not packet:
            raise ConnectionError("Socket connection closed.")

        data += packet

    return data


def receive_message(sock: socket.socket) -> Dict[str, Any]:
    header = recv_exact(sock, HEADER_SIZE)
    payload_size = int.from_bytes(header, byteorder="big")

    payload = recv_exact(sock, payload_size)

    return json.loads(payload.decode("utf-8"))


def send_message(sock: socket.socket, message: Dict[str, Any]) -> None:
    sock.sendall(encode_message(message))


def ok_response(**kwargs) -> Dict[str, Any]:
    response = {
        "status": "ok"
    }

    response.update(kwargs)
    return response


def error_response(message: str, **kwargs) -> Dict[str, Any]:
    response = {
        "status": "error",
        "message": message
    }

    response.update(kwargs)
    return response