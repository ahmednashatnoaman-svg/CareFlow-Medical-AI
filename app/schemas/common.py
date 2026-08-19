"""Standard CareFlow API Envelope schemas."""

from typing import Generic, Optional, TypeVar
from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """Standard successful API response envelope."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool = True
    message: str = "Operation completed successfully"
    data: Optional[T] = None


class ApiErrorResponse(BaseModel):
    """Standard error API response envelope."""

    success: bool = False
    error_code: str = "INTERNAL_SERVER_ERROR"
    message: str = "An unexpected error occurred"
