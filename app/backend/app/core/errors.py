from dataclasses import dataclass, field
from typing import Any


@dataclass
class ApiError(Exception):
    status_code: int
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    field_name: str | None = None


def error_payload(error: ApiError) -> dict[str, object]:
    payload: dict[str, object] = {
        "code": error.code,
        "message": error.message,
        "details": error.details,
    }
    if error.field_name is not None:
        payload["field"] = error.field_name
    return {"error": payload}
