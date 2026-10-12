"""Custom middleware for the service."""

import json
import logging
import time
import uuid

from typing import Any
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

# Polled by the container healthcheck and Prometheus; logging them would drown real traffic.
_QUIET_PATHS = ("/v1/healthz", "/metrics")


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured request/response logging."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Process request and log details."""
        if request.url.path.startswith(_QUIET_PATHS):
            return await call_next(request)
        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Add request ID to state
        request.state.request_id = request_id

        # Log request
        log_data = {
            "event": "request_started",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else None,
        }
        logger.info(json.dumps(log_data))

        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            logger.exception("Request failed")
            raise
        finally:
            # Log response
            duration_ms = (time.time() - start_time) * 1000
            log_data = {
                "event": "request_completed",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code if "response" in locals() else 500,
                "duration_ms": duration_ms,
            }
            logger.info(json.dumps(log_data))

        return response