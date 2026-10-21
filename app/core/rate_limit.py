import time

from fastapi import Request

from app.core.errors import AppException
from app.core.logging import get_logger
from app.core.redis import get_redis_client

logger = get_logger(__name__)


class RateLimiter:
    """Redis-backed sliding window rate limiter with in-memory fallback.

    Ensures endpoints are protected against brute-force attacks and abuse.
    """

    _redis_offline_until: float = 0.0

    def __init__(
        self,
        max_requests: int = 60,
        window_seconds: int = 60,
        prefix: str = "rate_limit",
    ) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.prefix = prefix
        # In-memory timestamp list fallback for standalone zero-Docker environments
        self._memory_buckets: dict[str, list[float]] = {}

    def reset(self) -> None:
        """Reset internal rate limit tracking state (useful for test isolation)."""
        self._memory_buckets.clear()

    def _get_client_identifier(self, request: Request) -> str:
        """Derive client IP or forwarded host as rate limit bucket identifier."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        client_host = request.client.host if request.client else "127.0.0.1"
        return client_host

    async def __call__(self, request: Request) -> None:
        """Execute rate limit check for incoming HTTP request."""
        client_id = self._get_client_identifier(request)
        key = f"{self.prefix}:{client_id}"

        # 1. Attempt Redis-backed distributed rate limiting (if not recently failed)
        now = time.time()
        if now >= RateLimiter._redis_offline_until:
            try:
                redis = get_redis_client()
                count = await redis.incr(key)
                if count == 1:
                    await redis.expire(key, self.window_seconds)

                if count > self.max_requests:
                    ttl = await redis.ttl(key)
                    logger.warning(f"Rate limit exceeded for {client_id} on {self.prefix}: {count}/{self.max_requests}")
                    raise AppException(
                        message=f"Rate limit exceeded. Maximum {self.max_requests} requests per {self.window_seconds}s.",
                        code="RATE_LIMIT_EXCEEDED",
                        status_code=429,
                        details={"retry_after_seconds": max(ttl, 1), "limit": self.max_requests},
                    )
                return
            except AppException:
                raise
            except Exception as exc:
                RateLimiter._redis_offline_until = now + 10.0
                logger.debug(f"Redis rate limiter unavailable (circuit breaker engaged for 10s): {exc}")


        # 2. In-memory sliding window fallback
        now = time.time()
        bucket = self._memory_buckets.get(key, [])
        # Prune expired timestamps
        bucket = [t for t in bucket if now - t < self.window_seconds]
        if len(bucket) >= self.max_requests:
            retry_after = int(self.window_seconds - (now - bucket[0]))
            logger.warning(f"Memory rate limit exceeded for {client_id} on {self.prefix}")
            raise AppException(
                message=f"Rate limit exceeded. Maximum {self.max_requests} requests per {self.window_seconds}s.",
                code="RATE_LIMIT_EXCEEDED",
                status_code=429,
                details={"retry_after_seconds": max(retry_after, 1), "limit": self.max_requests},
            )
        bucket.append(now)
        self._memory_buckets[key] = bucket
