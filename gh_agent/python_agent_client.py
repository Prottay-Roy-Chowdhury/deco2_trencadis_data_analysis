import os
import socket
from typing import Any, Dict
from pathlib import Path

from gh_agent.config import (
    PYTHON_AGENT_HOST,
    PYTHON_AGENT_PORT,
    PYTHON_AGENT_FILE_PORT,
    PYTHON_AGENT_UPLOAD_PORT,
    FILE_CHUNK_SIZE,
)
from gh_agent.protocol import send_message, receive_message 


class PythonAgentClient:
    def __init__(
            self,
            host: str = PYTHON_AGENT_HOST,
            port: int = PYTHON_AGENT_PORT):
        self.host = host
        self.port = port

    def send_command(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.host, self.port))

            send_message(sock, payload)

            response = receive_message(sock)

        return response
    
    def download_file(self, remote_file_path, local_file_path):
        
        local_file_path = Path(local_file_path)
        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.host, PYTHON_AGENT_FILE_PORT))

            send_message(sock, {
                "file_path": remote_file_path
            })

            header = receive_message(sock)

            if header.get("status") != "ok":
                return header

            file_size = int(header["file_size"])
            received = 0

            with open(local_file_path, "wb") as f:
                while received < file_size:
                    chunk = sock.recv(min(FILE_CHUNK_SIZE, file_size - received))
                    if not chunk:
                        break

                    f.write(chunk)
                    received += len(chunk)

            return {
                "status": "ok",
                "local_path": str(local_file_path),
                "file_size": file_size,
                "received": received
            }
        
        def upload_design_output(
            self,
            session,
            design_output_index,
            files,
            source_processing_output_index=None,
            created_by="design_pc",
            message="",
        ):
            """
            Uploads local design files to the Python master.

            files format:

                [
                    {
                        "path": "C:/.../solution.ghdata",
                        "category": "geometry",
                    },
                    {
                        "path": "C:/.../parameters.json",
                        "category": "parameters",
                    },
                ]
            """
            if not session:
                raise ValueError("Session cannot be empty.")

            try:
                design_index = int(design_output_index)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "design_output_index must be an integer."
                ) from exc

            if design_index < 1:
                raise ValueError(
                    "design_output_index must be at least 1."
                )

            if not isinstance(files, list) or not files:
                raise ValueError(
                    "files must be a non-empty list."
                )

            prepared_files = []

            for index, file_record in enumerate(files):
                if not isinstance(file_record, dict):
                    raise ValueError(
                        f"files[{index}] must be an object."
                    )

                local_path = Path(
                    file_record.get("path") or ""
                )

                category = str(
                    file_record.get("category") or ""
                ).strip()

                if not local_path.is_file():
                    raise FileNotFoundError(
                        f"Design file does not exist: {local_path}"
                    )

                if not category:
                    raise ValueError(
                        f"files[{index}] is missing category."
                    )

                prepared_files.append(
                    {
                        "path": local_path,
                        "name": local_path.name,
                        "category": category,
                        "size": local_path.stat().st_size,
                    }
                )

            request = {
                "command": "upload_design_output",
                "session": str(session).strip(),
                "design_output_index": design_index,
                "created_by": str(created_by or "design_pc"),
                "message": str(message or ""),
                "files": [
                    {
                        "name": item["name"],
                        "category": item["category"],
                        "size": item["size"],
                    }
                    for item in prepared_files
                ],
            }

            if source_processing_output_index is not None:
                request["source_processing_output_index"] = int(
                    source_processing_output_index
                )

            with socket.socket(
                socket.AF_INET,
                socket.SOCK_STREAM,
            ) as sock:
                sock.connect(
                    (
                        self.host,
                        PYTHON_AGENT_UPLOAD_PORT,
                    )
                )

                send_message(sock, request)

                ready_response = receive_message(sock)

                if ready_response.get("status") != "ok":
                    return ready_response

                for item in prepared_files:
                    with item["path"].open("rb") as file:
                        while True:
                            chunk = file.read(FILE_CHUNK_SIZE)

                            if not chunk:
                                break

                            sock.sendall(chunk)

                completion_response = receive_message(sock)

                return completion_response