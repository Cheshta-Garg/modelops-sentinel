# Phase 05 — Authentication, Role-Based Access Control (RBAC), Audit Logging & Rate Limiting

## Objective
Establish a production-realistic, interview-defensible security foundation for **ModelOps Sentinel**.
The objective is to strictly govern model lifecycle transitions, protect production inference routes, ensure non-repudiable audit trails, and shield authentication endpoints from automated brute-force attacks without over-engineering into a generic enterprise IAM platform.

---

## Why This Stage Exists
Machine learning platforms deployed in production face unique security vulnerabilities if left unprotected:
1. **Model Tampering & Supply-Chain Attacks**: Without authentication and strict authorization, any internal actor or compromised network entity could register untrusted weights, overwrite production versions, trigger rogue canary rollouts, or initiate destructive rollbacks.
2. **Missing Accountability (Non-Repudiation)**: When an inference anomaly or production outage occurs, operations teams must trace: *Who registered this model?*, *Who uploaded this artifact?*, *When was it deployed?*, and *What was the exact state before/after?*
3. **Automated Credential Abuse & Resource Exhaustion**: Publicly accessible login endpoints are primary targets for automated credential stuffing and denial-of-service attacks.
4. **Separation of Concerns (Least Privilege)**: Data scientists/engineers require model creation, version uploading, and deployment privileges, whereas monitoring dashboards, executive viewers, and downstream consumers should only have read and inference access. Administrators maintain emergency oversight.

---

## Core Theoretical Concepts Explained

### 1. Authentication vs Authorization
- **Authentication (AuthN)** answers: *"Who are you?"*
  - It is the mechanism of verifying identity (e.g. verifying that a client presenting email `engineer@modelops.io` and password `Password123!` actually owns the account).
  - In ModelOps Sentinel, authentication is performed via `POST /api/v1/auth/login`, which returns a signed, time-bounded JSON Web Token (JWT).
- **Authorization (AuthZ)** answers: *"What are you allowed to do?"*
  - It determines whether an authenticated entity possesses the required permissions to execute a specific action on a protected resource.
  - In ModelOps Sentinel, authorization is enforced via FastAPI dependencies (`require_roles(...)`), checking whether the authenticated user's role satisfies the endpoint's privilege constraints.

```
+-----------------------------------------------------------------------------------+
| Request -> [Authentication: Who is this?] -> [Authorization: Is this role allowed?] |
|             - Verify JWT signature             - Check user.role in allowed_roles |
|             - Decode subject UUID              - Pass or raise 403 Forbidden      |
+-----------------------------------------------------------------------------------+
```

### 2. JSON Web Tokens (JWT)
A **JSON Web Token (JWT)** (RFC 7519) is a compact, URL-safe, stateless credential format consisting of three base64url-encoded parts separated by dots: `Header.Payload.Signature`.
- **Header**: Contains metadata about the algorithm and token type (`{"alg": "HS256", "typ": "JWT"}`).
- **Payload**: Contains claims—statements about the user and token expiration:
  - `sub`: The immutable user UUID.
  - `email`: User email address.
  - `role`: The assigned RBAC role (`admin`, `engineer`, `viewer`).
  - `exp`: Expiration Unix timestamp (configured to 24 hours).
  - `iat`: Issued-at Unix timestamp.
- **Signature**: Formed by cryptographically signing `base64url(Header) + "." + base64url(Payload)` using the server's private secret key and HMAC-SHA256 (`HS256`).
- **Stateless Verification**: The API server validates incoming tokens simply by recomputing the signature against its local secret key. This eliminates database lookup overhead on every authenticated request while enabling horizontal scaling across API replicas.

### 3. Password Hashing (Bcrypt & Salting)
Passwords must **never** be stored in plaintext or with reversible encryption. ModelOps Sentinel uses the industry-standard **Bcrypt** adaptive hashing algorithm:
- **Cryptographic Salting**: Bcrypt automatically generates a unique 128-bit random salt per password before hashing. This ensures identical plaintext passwords yield completely distinct hashes, defeating precomputed rainbow table attacks.
- **Work Factor (Cost Factor = 12)**: Bcrypt utilizes a parameterizable key derivation function ($2^{12} = 4096$ iterations). This intentional CPU cost renders massive GPU-accelerated dictionary attacks infeasible while remaining imperceptible (~150ms) to legitimate users logging in once.
- **Constant-Time Verification**: Verifications utilize `bcrypt.checkpw()`, which compares hashes in constant time to eliminate side-channel timing attack vulnerabilities.

### 4. Role-Based Access Control (RBAC)
**Role-Based Access Control (RBAC)** restricts system access based on assigned organizational responsibilities rather than ad-hoc per-user permissions.
- **Principle of Least Privilege**: Users are granted only the minimum permissions necessary to perform their job duties.
- **Hierarchical Governance**:
  - `viewer`: Read-only access to catalog models, model details, metadata, and inference execution.
  - `engineer`: Lifecycle operations—creating models, uploading version artifacts, initiating deployments, and executing rollbacks.
  - `admin`: Universal supervisor access across all platform entities and operational controls.

### 5. Access Tokens & The Bearer Scheme
An **access token** is an opaque or structured credential presented by the client to access protected resources without repeatedly transmitting raw passwords.
- **Bearer HTTP Scheme**: The client transmits the token in the standard HTTP header:
  ```http
  Authorization: Bearer <access_token>
  ```
- **Lifecycle & Expiry**: Tokens are configured with a strict expiration window (`ACCESS_TOKEN_EXPIRE_MINUTES = 1440`). Once expired, requests fail with `401 Unauthorized` (`TOKEN_EXPIRED`), preventing indefinite access if a token is inadvertently intercepted.

### 6. Rate Limiting
Rate limiting protects system availability and integrity by capping the frequency of requests an IP address or client identifier can submit within a rolling time window.
- **Security Purpose**: Prevents brute-force dictionary attacks against `/auth/login`, automated registration spam on `/auth/register`, and denial-of-service degradation.
- **Sliding Window Algorithm**: Rather than a crude fixed-window counter that resets abruptly at the top of the minute, a sliding window evaluates the request density over the exact preceding $N$ seconds.
- **Redis-Backed with Local Fallback**: Rate counts are atomically managed in Redis using `INCR` and `EXPIRE`. If Redis is offline, a local sliding timestamp queue acts as an in-memory fallback, bolstered by a 10-second circuit breaker to prevent latency degradation. Excess requests receive `429 Too Many Requests` with a `Retry-After` header.

---

## RBAC Permission Matrix

| Endpoint | Method | Permitted Roles | Action Performed | Audit Logged? |
| :--- | :---: | :---: | :--- | :---: |
| `/api/v1/auth/register` | `POST` | Public (Rate-limited) | Account creation | Yes (`USER_REGISTERED`) |
| `/api/v1/auth/login` | `POST` | Public (Rate-limited) | Authenticate & issue JWT | Yes (`USER_LOGGED_IN`) |
| `/api/v1/auth/me` | `GET` | All authenticated | Read caller profile | No |
| `/api/v1/models` | `POST` | `ENGINEER`, `ADMIN` | Register logical model identity | Yes (`MODEL_CREATED`) |
| `/api/v1/models` | `GET` | `VIEWER`, `ENGINEER`, `ADMIN` | List registered models | No |
| `/api/v1/models/{id}` | `GET` | `VIEWER`, `ENGINEER`, `ADMIN` | Get model detail & versions | No |
| `/api/v1/models/{id}/versions` | `POST` | `ENGINEER`, `ADMIN` | Upload artifact & transition to STAGING | Yes (`MODEL_VERSION_UPLOADED`) |
| `/api/v1/models/{id}/predict` | `POST` | `VIEWER`, `ENGINEER`, `ADMIN` | Execute synchronous ML inference | Recorded in `inference_requests` |

---

## Architectural Workflows

### Authentication & Token Issuance Flow

```
Client                             FastAPI (/auth/login)              PostgreSQL
  │                                         │                              │
  ├─── POST /api/v1/auth/login ────────────▶│                              │
  │    {email, password}                    ├─▶ RateLimiter.check()        │
  │                                         ├─▶ Query user by email ──────▶│
  │                                         │◀─ Return User entity ────────┤
  │                                         ├─▶ bcrypt.checkpw()           │
  │                                         ├─▶ AuditService.log_action() ─▶│ (USER_LOGGED_IN)
  │                                         ├─▶ create_access_token()      │
  │◀── 200 OK {access_token, token_type} ───┤                              │
```

### Protected Request with RBAC & Audit Trail

```
Client                        FastAPI (require_roles)          Registry/Serving         Database
  │                                      │                            │                     │
  ├─── POST /api/v1/models ─────────────▶│                            │                     │
  │    Header: Bearer <jwt>              ├─▶ decode_access_token()    │                     │
  │                                      ├─▶ Resolve user from DB ─────────────────────────▶│
  │                                      ├─▶ Check role in allowed?   │                     │
  │                                      │   (viewer -> 403 Forbidden)│                     │
  │                                      │   (engineer/admin -> OK)   │                     │
  │                                      └────────────┬──────────────▶│                     │
  │                                                   │               ├─▶ Create Model ────▶│
  │                                                   │               ├─▶ Log Audit Event ─▶│ (MODEL_CREATED)
  │◀── 201 Created {model_id, name} ──────────────────┴───────────────┤                     │
```

---

## What Was Implemented

1. **Security Core (`app/core/security.py`)**:
   - `hash_password(raw_password: str) -> str`: Uses `bcrypt` with cost factor 12 to generate cryptographic hashes.
   - `verify_password(plain: str, hashed: str) -> bool`: Safe verification using `bcrypt.checkpw()`.
   - `create_access_token(user_id, email, role) -> str`: Encodes standard claims (`sub`, `email`, `role`, `exp`, `iat`) signed via `HS256`.
   - `decode_access_token(token: str) -> dict[str, Any]`: Validates signature and expiration; raises structured `AppException` on malformed or expired tokens.
2. **Authentication & RBAC Dependencies (`app/core/auth.py`)**:
   - `get_current_user`: Extracts Bearer token, decodes claims, verifies user existence and active status in the database.
   - `require_roles(*allowed_roles)`: Factory yielding dependency functions. Enforces role checks with `ADMIN` possessing full universal access.
3. **Redis Sliding-Window Rate Limiter (`app/core/rate_limit.py`)**:
   - Atomic distributed counter with Redis TTL.
   - Sliding timestamp memory fallback for standalone/zero-Docker environments.
   - 10-second circuit breaker preventing latency spikes when Redis is unreachable.
4. **Transactional Audit Logging Service (`app/services/audit.py`)**:
   - Immutable persistence of critical state changes into the `audit_logs` table (`USER_REGISTERED`, `USER_LOGGED_IN`, `MODEL_CREATED`, `MODEL_VERSION_UPLOADED`).
5. **Authentication Endpoints (`app/api/v1/endpoints/auth.py`)**:
   - `POST /api/v1/auth/register`: Account creation with email format validation and uniqueness check.
   - `POST /api/v1/auth/login`: Credential validation and JWT issuance.
   - `GET /api/v1/auth/me`: Authenticated user profile lookup.
6. **Protected Model Endpoints (`app/api/v1/endpoints/models.py`)**:
   - Attached `require_roles(UserRole.ENGINEER, UserRole.ADMIN)` to model creation and artifact upload routes.
   - Attached `require_roles(UserRole.VIEWER, UserRole.ENGINEER, UserRole.ADMIN)` to model viewing and inference routes.
   - Integrated automatic audit logging on successful model registration and version upload.

---

## Files Created / Modified

| File | Type | Description |
| :--- | :---: | :--- |
| `app/models/enums.py` | Modified | Added `UserRole` (`admin`, `engineer`, `viewer`, `operator`) with case-insensitive resolution. |
| `app/core/config.py` | Modified | Added `JWT_ALGORITHM` ("HS256") and `ACCESS_TOKEN_EXPIRE_MINUTES` (1440). |
| `app/core/security.py` | Created | Bcrypt password hashing and PyJWT token generation/validation. |
| `app/schemas/auth.py` | Created | Pydantic DTOs for register, login, user response, and token response. |
| `app/services/audit.py` | Created | Transactional audit trail logging service writing to `audit_logs`. |
| `app/core/rate_limit.py` | Created | Redis-backed sliding window rate limiter with in-memory fallback. |
| `app/core/auth.py` | Created | FastAPI authentication and RBAC dependency injectors. |
| `app/services/auth.py` | Created | Business logic for registration, authentication, and credential validation. |
| `app/api/v1/endpoints/auth.py` | Created | HTTP route handlers for `/auth/register`, `/auth/login`, and `/auth/me`. |
| `app/api/v1/router.py` | Modified | Registered `auth.router` into API router v1. |
| `app/services/registry.py` | Modified | Updated model and version creation to associate creator `created_by_id`. |
| `app/api/v1/endpoints/models.py` | Modified | Protected endpoints with `require_roles` and added audit event logging. |
| `tests/conftest.py` | Modified | Added `DEFAULT_TEST_ADMIN` dependency override on `async_client` and `unauthenticated_client`. |
| `tests/test_auth.py` | Created | 12 automated unit and integration tests covering auth, RBAC, and rate limiting. |
| `scripts/test_stage5_manual.py` | Created | End-to-end zero-Docker manual verification script demonstrating full lifecycle. |
| `scripts/test_stage4_manual.py` | Modified | Updated with authenticated engineer token headers. |
| `docs/07_phase_05_auth.md` | Created | Architecture, theory, and operational manual for Phase 05. |

---

## Verification & Test Results

### 1. Automated Test Suite (`pytest`)
All 49 unit and integration tests pass across the entire platform:
```bash
.\.venv\Scripts\python.exe -m pytest -v
```
Output:
```
tests/test_auth.py::test_register_user_success PASSED                    [  2%]
tests/test_auth.py::test_register_duplicate_email_conflict PASSED        [  4%]
tests/test_auth.py::test_register_short_password_rejected PASSED         [  6%]
tests/test_auth.py::test_login_success PASSED                            [  8%]
tests/test_auth.py::test_login_invalid_password_returns_401 PASSED       [ 10%]
tests/test_auth.py::test_get_current_user_profile_via_token PASSED       [ 12%]
tests/test_auth.py::test_protected_endpoint_without_token_returns_401 PASSED [ 14%]
tests/test_auth.py::test_protected_endpoint_with_invalid_token_returns_401 PASSED [ 16%]
tests/test_auth.py::test_rbac_viewer_forbidden_from_creating_model PASSED [ 18%]
tests/test_auth.py::test_rbac_engineer_can_create_model PASSED           [ 20%]
tests/test_auth.py::test_rbac_admin_has_full_access PASSED               [ 22%]
tests/test_auth.py::test_rate_limiter_blocks_excessive_auth_requests PASSED [ 24%]
tests/test_database.py::... PASSED                                       [ 44%]
tests/test_health.py::... PASSED                                         [ 59%]
tests/test_inference.py::... PASSED                                      [ 79%]
tests/test_registry.py::... PASSED                                       [100%]
======================= 49 passed in 53.68s =======================
```

### 2. Manual Zero-Docker Verification Script
Run the automated end-to-end manual verification script:
```bash
.\.venv\Scripts\python.exe scripts/test_stage5_manual.py
```
Summary of Verified Behaviors:
- Registered `admin@modelops.io`, `engineer@modelops.io`, `viewer@modelops.io`.
- Verified Bcrypt hash in SQLite `users` table: starts with `$2b$12$`, plaintext password never saved.
- Verified `/auth/me` profile decoding.
- Verified missing token -> `401 UNAUTHENTICATED`.
- Verified tampered token -> `401 INVALID_TOKEN_SIGNATURE`.
- Verified Viewer attempting to create model -> `403 FORBIDDEN_INSUFFICIENT_PERMISSIONS`.
- Verified Engineer creating model & uploading artifact -> `201 Created`.
- Verified Viewer reading models & executing inference -> `200 OK`.
- Verified Admin universal access -> `200 OK`.
- Verified `audit_logs` table records (`USER_REGISTERED`, `USER_LOGGED_IN`, `MODEL_CREATED`, `MODEL_VERSION_UPLOADED`).
- Verified Rate Limiter bursts -> Request #11 blocked with `429 Too Many Requests` (`RATE_LIMIT_EXCEEDED`, `Retry-After: 60s`).

---

## Interview Defense / Technical Questions & Answers

### Q1: Why use JWT instead of standard server-side sessions with a session cookie?
**Answer**:
In high-throughput ML serving systems, stateless JWTs allow horizontally scaled API replicas to validate client authenticity and roles purely through CPU signature verification without querying a central session database on every single incoming prediction request. This prevents the session database from becoming a single point of failure or latency bottleneck. For token revocation, short expiration windows (or Redis-backed denylists for high-risk accounts) balance scalability with security.

### Q2: Why choose Bcrypt over SHA-256 or MD5 for password storage?
**Answer**:
SHA-256 and MD5 are general-purpose cryptographic hash functions designed to be extremely fast. Modern GPUs can calculate billions of SHA-256 hashes per second, making offline brute-force and dictionary attacks trivial. Bcrypt is deliberately slow and computationally intensive, incorporating an adaptive cost factor and automatic salt generation. A cost factor of 12 forces an attacker to spend ~150ms per guess, making dictionary attacks mathematically intractable.

### Q3: What happens if Redis goes offline during rate limiting?
**Answer**:
ModelOps Sentinel implements a resilient, fail-open rate limiting architecture:
1. When a Redis connectivity error occurs, a 10-second circuit breaker engages immediately.
2. Subsequent requests within the next 10 seconds fall back directly to an in-memory sliding window bucket, avoiding repeated socket connection timeout delays.
3. System operations remain protected against abuse without crashing user-facing APIs.
