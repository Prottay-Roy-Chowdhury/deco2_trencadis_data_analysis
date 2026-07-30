import os
import socket
from typing import Any, Dict
from pathlib import Path

from gh_agent.config import (
    PYTHON_AGENT_HOST,
    PYTHON_AGENT_PORT,
    PYTHON_AGENT_FILE_PORT,
    PYTHON_AGENT_UPLOAD_PORT,
    PYTHON_AGENT_MOTION_UPLOAD_PORT,
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

    def get_pending_project_action(
        self,
        target,
        session=None,
    ):
        """
        Requests the oldest unclaimed project action assigned
        to this PC type.

        target examples:
            design_pc
            motion_pc

        session is optional. When omitted, the master searches
        all persisted project pipelines.
        """

        normalized_target = str(
            target or ""
        ).strip().lower()

        if not normalized_target:
            raise ValueError(
                "target cannot be empty."
            )

        payload = {
            "command": (
                "get_pending_project_action"
            ),
            "target": normalized_target,
        }

        normalized_session = str(
            session or ""
        ).strip()

        if normalized_session:
            payload["session"] = (
                normalized_session
            )

        return self.send_command(
            payload
        )

    def claim_project_action(
        self,
        session,
        action_id,
        claimed_by,
    ):
        """
        Claims one pending project action on the master.

        A successful claim changes the master pipeline from:

            design_pending -> design_requested

        or:

            motion_pending -> motion_requested
        """

        normalized_session = str(
            session or ""
        ).strip()

        normalized_action_id = str(
            action_id or ""
        ).strip()

        normalized_claimed_by = str(
            claimed_by or ""
        ).strip().lower()

        if not normalized_session:
            raise ValueError(
                "session cannot be empty."
            )

        if not normalized_action_id:
            raise ValueError(
                "action_id cannot be empty."
            )

        if not normalized_claimed_by:
            raise ValueError(
                "claimed_by cannot be empty."
            )

        payload = {
            "command": (
                "claim_project_action"
            ),
            "session": normalized_session,
            "action_id": normalized_action_id,
            "claimed_by": (
                normalized_claimed_by
            ),
        }

        return self.send_command(
            payload
        )

    def report_project_action(
        self,
        session,
        action_id,
        action_status,
        message="",
        result=None,
        error=None,
    ):
        """
        Reports the local execution state of a claimed action.

        Supported action_status values:

            running
            completed
            failed
            cancelled

        A completed report does not replace the master output
        manifest as durable stage-completion evidence.
        """

        normalized_session = str(
            session or ""
        ).strip()

        normalized_action_id = str(
            action_id or ""
        ).strip()

        normalized_status = str(
            action_status or ""
        ).strip().lower()

        valid_statuses = {
            "running",
            "completed",
            "failed",
            "cancelled",
        }

        if not normalized_session:
            raise ValueError(
                "session cannot be empty."
            )

        if not normalized_action_id:
            raise ValueError(
                "action_id cannot be empty."
            )

        if normalized_status not in valid_statuses:
            raise ValueError(
                "action_status must be one of: "
                + ", ".join(
                    sorted(
                        valid_statuses
                    )
                )
                + "."
            )

        payload = {
            "command": (
                "report_project_action"
            ),
            "session": normalized_session,
            "action_id": normalized_action_id,
            "action_status": (
                normalized_status
            ),
            "message": str(
                message or ""
            ),
        }

        if result is not None:
            payload["result"] = result

        if error is not None:
            payload["error"] = error

        return self.send_command(
            payload
        )

    def get_design_project_action(
        self,
        session=None,
    ):
        return self.get_pending_project_action(
            target="design_pc",
            session=session,
        )

    def get_motion_project_action(
        self,
        session=None,
    ):
        return self.get_pending_project_action(
            target="motion_pc",
            session=session,
        )
    
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
        *,
        solution_index,
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

                try:
                    solution_index = int(solution_index)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "solution_index must be an integer."
                    ) from exc

                if solution_index < 1:
                    raise ValueError(
                        "solution_index must be at least 1."
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
                    "solution_index": solution_index,
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

    def upload_motion_output(
        self,
        session,
        motion_output_index,
        files,
        *,
        solution_index,
        source_design_output_index=None,
        created_by="motion_pc",
        message="",
    ):
        """
        Uploads local motion files to the Python master.

        files format:

        [
            {
                "path": "C:/.../motion_ghdata_01.ghdata",
                "category": "ghdata",
            },
            {
                "path": "C:/.../robot_program.mod",
                "category": "program",
            },
        ]
        """

        if not session:
            raise ValueError(
                "Session cannot be empty."
            )

        try:
            motion_index = int(
                motion_output_index
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "motion_output_index must be an integer."
            ) from exc

        if motion_index < 1:
            raise ValueError(
                "motion_output_index must be at least 1."
            )

        source_design_index = None

        if source_design_output_index is not None:
            try:
                source_design_index = int(
                    source_design_output_index
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "source_design_output_index "
                    "must be an integer."
                ) from exc

            if source_design_index < 1:
                raise ValueError(
                    "source_design_output_index "
                    "must be at least 1."
                )

        if not isinstance(files, list) or not files:
            raise ValueError(
                "files must be a non-empty list."
            )

        try:
            solution_index = int(solution_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "solution_index must be an integer."
            ) from exc

        if solution_index < 1:
            raise ValueError(
                "solution_index must be at least 1."
            )

        prepared_files = []

        for index, file_record in enumerate(files):
            if not isinstance(file_record, dict):
                raise ValueError(
                    f"files[{index}] must be an object."
                )

            local_path = Path(
                file_record.get("path") or ""
            ).expanduser()

            category = str(
                file_record.get("category") or ""
            ).strip()

            if not local_path.is_file():
                raise FileNotFoundError(
                    "Motion file does not exist: "
                    f"{local_path}"
                )

            if not category:
                raise ValueError(
                    f"files[{index}] is missing category."
                )

            prepared_files.append(
                {
                    "path": local_path.resolve(),
                    "name": local_path.name,
                    "category": category,
                    "size": local_path.stat().st_size,
                }
            )

        request = {
            "command": "upload_motion_output",
            "session": str(session).strip(),
            "motion_output_index": motion_index,
            "solution_index": solution_index,
            "created_by": str(
                created_by or "motion_pc"
            ).strip(),
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

        if source_design_index is not None:
            request[
                "source_design_output_index"
            ] = source_design_index

        with socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        ) as sock:
            sock.connect(
                (
                    self.host,
                    PYTHON_AGENT_MOTION_UPLOAD_PORT,
                )
            )

            send_message(
                sock,
                request,
            )

            ready_response = receive_message(
                sock
            )

            if ready_response.get("status") != "ok":
                return ready_response

            for item in prepared_files:
                with item["path"].open("rb") as file:
                    while True:
                        chunk = file.read(
                            FILE_CHUNK_SIZE
                        )

                        if not chunk:
                            break

                        sock.sendall(chunk)

            completion_response = receive_message(
                sock
            )

            return completion_response