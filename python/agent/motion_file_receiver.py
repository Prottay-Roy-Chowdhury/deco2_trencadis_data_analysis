import os
import socket
import traceback

from collections import defaultdict
from pathlib import Path
from typing import Any

from agent.config import (
    FILE_CHUNK_SIZE,
    HOST,
    PROJECT_ROOT,
    UPLOAD_PORT,
)
from agent.protocol import (
    error_response,
    receive_message,
    send_message,
)
from helpers.motion_output_store import MotionOutputStore


class MotionFileReceiver:
    """
    Receives motion files uploaded from a GH Agent.

    Protocol for one connection:

    1. Client sends one framed JSON upload request.
    2. Server validates the request and prepares the manifest entry.
    3. Server sends a framed JSON ready response.
    4. Client streams each file as raw bytes, in the same order
       as the request's files array.
    5. Server sends one framed JSON completion response.

    Each file record must include:

        name
        size
        category

    The raw byte stream contains exactly `size` bytes for each file.
    """

    def __init__(
        self,
        host: str = HOST,
        port: int = UPLOAD_PORT,
        sessions_root: Path | None = None,
    ):
        self.host = host
        self.port = port

        if sessions_root is None:
            sessions_root = (
                PROJECT_ROOT /
                "sessions"
            )

        self.store = MotionOutputStore(
            sessions_root=sessions_root,
        )

        self.running = False

    def start(self) -> None:
        print(
            "[motion-file-receiver] "
            f"Listening on {self.host}:{self.port}"
        )

        self.running = True

        with socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        ) as server:
            server.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_REUSEADDR,
                1,
            )

            server.bind(
                (
                    self.host,
                    self.port,
                )
            )

            server.listen()
            server.settimeout(1.0)

            while self.running:
                try:
                    client, address = (
                        server.accept()
                    )
                except socket.timeout:
                    continue

                with client:
                    print(
                        "[motion-file-receiver] "
                        f"Connected: {address}"
                    )

                    try:
                        self._handle_client(
                            client
                        )
                    except Exception:
                        traceback.print_exc()

                        try:
                            send_message(
                                client,
                                error_response(
                                    "Motion upload receiver "
                                    "exception."
                                ),
                            )
                        except Exception:
                            pass

    def stop(self) -> None:
        self.running = False

    def _handle_client(
        self,
        client: socket.socket,
    ) -> None:
        request = receive_message(
            client
        )

        if (
            request.get("command") !=
            "upload_motion_output"
        ):
            send_message(
                client,
                error_response(
                    "Unsupported upload command."
                ),
            )
            return

        session = request.get(
            "session"
        )

        motion_output_index = request.get(
            "motion_output_index"
        )

        source_design_output_index = (
            request.get(
                "source_design_output_index"
            )
        )

        created_by = request.get(
            "created_by",
            "motion_pc",
        )

        message = request.get(
            "message",
            "",
        )

        files = request.get(
            "files"
        )

        if not session:
            send_message(
                client,
                error_response(
                    "Missing session."
                ),
            )
            return

        if motion_output_index is None:
            send_message(
                client,
                error_response(
                    "Missing motion_output_index."
                ),
            )
            return

        if (
            not isinstance(files, list) or
            not files
        ):
            send_message(
                client,
                error_response(
                    "files must be a "
                    "non-empty array."
                ),
            )
            return

        validated_files = (
            self._validate_files(
                files
            )
        )

        upload_state = (
            self.store.begin_upload(
                session=session,
                motion_output_index=(
                    motion_output_index
                ),
                source_design_output_index=(
                    source_design_output_index
                ),
                message=message,
                created_by=created_by,
            )
        )

        output_folder = Path(
            upload_state[
                "output_folder"
            ]
        )

        send_message(
            client,
            {
                "status": "ok",
                "message": (
                    "Ready to receive "
                    "motion files."
                ),
                "session": session,
                "motion_output_index": int(
                    motion_output_index
                ),
                "file_count": len(
                    validated_files
                ),
            },
        )

        stored_records: list[
            dict[str, Any]
        ] = []

        try:
            category_counts: dict[
                str,
                int,
            ] = defaultdict(int)

            total_per_category: dict[
                str,
                int,
            ] = defaultdict(int)

            for file_info in validated_files:
                total_per_category[
                    file_info["category"]
                ] += 1

            for file_info in validated_files:
                category = file_info[
                    "category"
                ]

                category_counts[
                    category
                ] += 1

                category_sequence = None

                if (
                    total_per_category[
                        category
                    ] > 1
                ):
                    category_sequence = (
                        category_counts[
                            category
                        ]
                    )

                stored_filename = (
                    self.store
                    .build_stored_filename(
                        category=category,
                        motion_output_index=(
                            motion_output_index
                        ),
                        original_filename=(
                            file_info["name"]
                        ),
                        category_sequence=(
                            category_sequence
                        ),
                    )
                )

                destination = (
                    output_folder /
                    stored_filename
                )

                self._receive_file(
                    client=client,
                    destination=destination,
                    expected_size=(
                        file_info["size"]
                    ),
                )

                stored_records.append(
                    {
                        "category": category,
                        "filename": (
                            stored_filename
                        ),
                        "original_filename": (
                            file_info["name"]
                        ),
                        "size_bytes": (
                            file_info["size"]
                        ),
                    }
                )

            completed_entry = (
                self.store.complete_upload(
                    session=session,
                    motion_output_index=(
                        motion_output_index
                    ),
                    files=stored_records,
                    message=message,
                )
            )

            send_message(
                client,
                {
                    "status": "ok",
                    "message": (
                        "Motion output uploaded "
                        "successfully."
                    ),
                    "session": session,
                    "motion_output_index": int(
                        motion_output_index
                    ),
                    "output": (
                        completed_entry
                    ),
                },
            )

            print(
                "[motion-file-receiver] "
                f"Completed session={session}, "
                "motion_output_index="
                f"{motion_output_index}"
            )

        except Exception as exc:
            self.store.fail_upload(
                session=session,
                motion_output_index=(
                    motion_output_index
                ),
                error=str(exc),
                files=stored_records,
            )

            raise

    def _validate_files(
        self,
        files: list[Any],
    ) -> list[dict[str, Any]]:
        validated: list[
            dict[str, Any]
        ] = []

        for index, file_info in enumerate(
            files
        ):
            if not isinstance(
                file_info,
                dict,
            ):
                raise ValueError(
                    f"files[{index}] must "
                    "be an object."
                )

            name = str(
                file_info.get("name") or ""
            ).strip()

            category = str(
                file_info.get(
                    "category"
                ) or ""
            ).strip()

            raw_size = file_info.get(
                "size"
            )

            if not name:
                raise ValueError(
                    f"files[{index}] is "
                    "missing name."
                )

            if Path(name).name != name:
                raise ValueError(
                    f"files[{index}].name must "
                    "be a filename, not a path."
                )

            if not category:
                raise ValueError(
                    f"files[{index}] is "
                    "missing category."
                )

            try:
                size = int(
                    raw_size
                )
            except (
                TypeError,
                ValueError,
            ) as exc:
                raise ValueError(
                    f"files[{index}].size "
                    "must be an integer."
                ) from exc

            if size < 0:
                raise ValueError(
                    f"files[{index}].size "
                    "cannot be negative."
                )

            validated.append(
                {
                    "name": name,
                    "category": category,
                    "size": size,
                }
            )

        return validated

    def _receive_file(
        self,
        client: socket.socket,
        destination: Path,
        expected_size: int,
    ) -> None:
        temporary_path = (
            destination.with_suffix(
                destination.suffix +
                ".uploading"
            )
        )

        received_size = 0

        try:
            with temporary_path.open(
                "wb"
            ) as file:
                while (
                    received_size <
                    expected_size
                ):
                    remaining = (
                        expected_size -
                        received_size
                    )

                    chunk = client.recv(
                        min(
                            FILE_CHUNK_SIZE,
                            remaining,
                        )
                    )

                    if not chunk:
                        raise ConnectionError(
                            "Connection closed "
                            "during upload of "
                            f"{destination.name}."
                        )

                    file.write(
                        chunk
                    )

                    received_size += len(
                        chunk
                    )

            if (
                received_size !=
                expected_size
            ):
                raise IOError(
                    "Incorrect upload size for "
                    f"{destination.name}: "
                    f"expected {expected_size}, "
                    f"received {received_size}."
                )

            os.replace(
                temporary_path,
                destination,
            )

        except Exception:
            if temporary_path.exists():
                temporary_path.unlink()

            raise