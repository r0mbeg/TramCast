-- name: LockPredictionJobAdmission :exec
-- Short READ COMMITTED transaction, BEFORE taking any row locks:
-- lock -> GetPredictionJobByRouteAndVersion -> CountPendingPredictionJobs
-- -> CreatePredictionJob if below capacity -> commit. Reuse existing jobs even
-- at capacity. The lock serializes admissions; never keep it during an RPC.
LOCK TABLE prediction_jobs IN SHARE ROW EXCLUSIVE MODE;

-- name: CountPendingPredictionJobs :one
-- Admission capacity covers both waiting and running jobs, including retries.
SELECT count(*) FROM prediction_jobs WHERE status IN ('queued', 'running');

-- name: CreatePredictionJob :one
-- Use only in the admission transaction. On conflict, no row is returned:
-- issue GetPredictionJobByRouteAndVersion as a NEW statement (fresh snapshot).
-- Do not combine INSERT DO NOTHING and fallback SELECT in one CTE snapshot.
INSERT INTO prediction_jobs (id, forecast_version_id, route_id)
VALUES (
    sqlc.arg(id)::uuid,
    sqlc.arg(forecast_version_id)::uuid,
    sqlc.arg(route_id)::bigint
)
ON CONFLICT (forecast_version_id, route_id) DO NOTHING
RETURNING id, forecast_version_id, route_id, status, attempt_count, run_after,
    lease_until, created_at, started_at, finished_at, last_error_code, last_error_message;

-- name: GetPredictionJob :one
SELECT id, forecast_version_id, route_id, status, attempt_count, run_after,
    lease_until, created_at, started_at, finished_at, last_error_code, last_error_message
FROM prediction_jobs
WHERE id = sqlc.arg(id)::uuid;

-- name: GetPredictionJobByRouteAndVersion :one
SELECT id, forecast_version_id, route_id, status, attempt_count, run_after,
    lease_until, created_at, started_at, finished_at, last_error_code, last_error_message
FROM prediction_jobs
WHERE forecast_version_id = sqlc.arg(forecast_version_id)::uuid
    AND route_id = sqlc.arg(route_id)::bigint;

-- name: ClaimNextPredictionJob :one
-- Commit the short claim transaction BEFORE starting the RPC. The returned
-- attempt_count is the ownership token used by every worker mutation.
WITH candidate AS MATERIALIZED (
    SELECT id
    FROM prediction_jobs
    WHERE status = 'queued'
        AND run_after <= clock_timestamp()
        AND attempt_count < sqlc.arg(max_attempts)::integer
        AND sqlc.arg(lease_seconds)::integer > 0
    ORDER BY run_after, created_at, id
    LIMIT 1
    FOR UPDATE SKIP LOCKED
)
UPDATE prediction_jobs AS job
SET status = 'running',
    attempt_count = job.attempt_count + 1,
    started_at = clock_timestamp(),
    lease_until = clock_timestamp() + sqlc.arg(lease_seconds)::integer * INTERVAL '1 second',
    finished_at = NULL
FROM candidate
WHERE job.id = candidate.id
RETURNING job.id, job.forecast_version_id, job.route_id, job.status, job.attempt_count,
    job.run_after, job.lease_until, job.created_at, job.started_at, job.finished_at,
    job.last_error_code, job.last_error_message;

-- name: RenewPredictionJobLease :execrows
-- Materialization ensures ownership is checked against the locked row and the
-- actual clock AFTER waiting for that lock, not transaction-start time.
-- Zero affected rows means ownership was lost (or lease_seconds was invalid).
WITH locked_job AS MATERIALIZED (
    SELECT id, status, attempt_count, lease_until
    FROM prediction_jobs
    WHERE id = sqlc.arg(id)::uuid
    FOR UPDATE
)
UPDATE prediction_jobs AS job
SET lease_until = GREATEST(
    locked_job.lease_until,
    clock_timestamp() + sqlc.arg(lease_seconds)::integer * INTERVAL '1 second'
)
FROM locked_job
WHERE job.id = locked_job.id
    AND locked_job.status = 'running'
    AND locked_job.attempt_count = sqlc.arg(attempt_number)::integer
    AND locked_job.lease_until > clock_timestamp()
    AND sqlc.arg(lease_seconds)::integer > 0;

-- name: LockPredictionJobForPublication :one
-- Must run in the SAME transaction as CopyValidationPredictions and
-- CompletePredictionJob. No row means ownership was lost: ROLLBACK.
-- Check the exact response grid and metadata before opening the transaction.
WITH locked_job AS MATERIALIZED (
    SELECT id, forecast_version_id, route_id, status, attempt_count, run_after,
        lease_until, created_at, started_at, finished_at, last_error_code, last_error_message
    FROM prediction_jobs
    WHERE id = sqlc.arg(id)::uuid
    FOR UPDATE
)
SELECT id, forecast_version_id, route_id, status, attempt_count, run_after,
    lease_until, created_at, started_at, finished_at, last_error_code, last_error_message
FROM locked_job
WHERE status = 'running'
    AND attempt_count = sqlc.arg(attempt_number)::integer
    AND lease_until > clock_timestamp();

-- name: CompletePredictionJob :execrows
-- ONLY after LockPredictionJobForPublication and a successful full COPY in the
-- same transaction. Require exactly one affected row, otherwise ROLLBACK.
-- Do not recheck expiry here: the row lock has protected ownership throughout
-- publication, even if the lease elapsed while copying the verified points.
UPDATE prediction_jobs
SET status = 'succeeded',
    finished_at = clock_timestamp(),
    lease_until = NULL,
    last_error_code = NULL,
    last_error_message = NULL
WHERE id = sqlc.arg(id)::uuid
    AND status = 'running'
    AND attempt_count = sqlc.arg(attempt_number)::integer;

-- name: FinishPredictionJobWithError :one
-- A stale worker cannot requeue/fail a newer attempt or revive an expired lease.
-- retryable=false is terminal. A retry also requires remaining attempts.
WITH locked_job AS MATERIALIZED (
    SELECT id, status, attempt_count, lease_until
    FROM prediction_jobs
    WHERE id = sqlc.arg(id)::uuid
    FOR UPDATE
)
UPDATE prediction_jobs AS job
SET status = CASE
        WHEN sqlc.arg(retryable)::boolean AND locked_job.attempt_count < sqlc.arg(max_attempts)::integer
        THEN 'queued' ELSE 'failed' END,
    run_after = CASE
        WHEN sqlc.arg(retryable)::boolean AND locked_job.attempt_count < sqlc.arg(max_attempts)::integer
        THEN clock_timestamp() + sqlc.arg(retry_delay_seconds)::integer * INTERVAL '1 second'
        ELSE job.run_after END,
    finished_at = CASE
        WHEN sqlc.arg(retryable)::boolean AND locked_job.attempt_count < sqlc.arg(max_attempts)::integer
        THEN NULL ELSE clock_timestamp() END,
    lease_until = NULL,
    last_error_code = sqlc.arg(error_code)::text,
    last_error_message = sqlc.arg(error_message)::text
FROM locked_job
WHERE job.id = locked_job.id
    AND locked_job.status = 'running'
    AND locked_job.attempt_count = sqlc.arg(attempt_number)::integer
    AND locked_job.lease_until > clock_timestamp()
    AND sqlc.arg(max_attempts)::integer > 0
    AND sqlc.arg(retry_delay_seconds)::integer >= 0
RETURNING job.id, job.forecast_version_id, job.route_id, job.status, job.attempt_count,
    job.run_after, job.lease_until, job.created_at, job.started_at, job.finished_at,
    job.last_error_code, job.last_error_message;

-- name: RecoverExpiredPredictionJobs :many
-- Bounded recovery; SKIP LOCKED leaves jobs being published/renewed alone.
-- The next claim, not recovery, increments attempt_count.
WITH expired_jobs AS MATERIALIZED (
    SELECT id
    FROM prediction_jobs
    WHERE status = 'running'
        AND lease_until <= clock_timestamp()
        AND sqlc.arg(max_attempts)::integer > 0
        AND sqlc.arg(retry_delay_seconds)::integer >= 0
    ORDER BY lease_until, id
    LIMIT sqlc.arg(batch_size)::integer
    FOR UPDATE SKIP LOCKED
)
UPDATE prediction_jobs AS job
SET status = CASE WHEN job.attempt_count < sqlc.arg(max_attempts)::integer
        THEN 'queued' ELSE 'failed' END,
    run_after = CASE WHEN job.attempt_count < sqlc.arg(max_attempts)::integer
        THEN clock_timestamp() + sqlc.arg(retry_delay_seconds)::integer * INTERVAL '1 second'
        ELSE job.run_after END,
    finished_at = CASE WHEN job.attempt_count < sqlc.arg(max_attempts)::integer
        THEN NULL ELSE clock_timestamp() END,
    lease_until = NULL,
    last_error_code = 'lease_expired',
    last_error_message = 'The worker lease expired before completion'
FROM expired_jobs
WHERE job.id = expired_jobs.id
RETURNING job.id, job.status, job.attempt_count;

-- name: FailExhaustedQueuedPredictionJobs :many
-- Run during worker maintenance, including after lowering max_attempts in
-- configuration, so exhausted queued jobs cannot occupy capacity forever.
WITH exhausted_jobs AS MATERIALIZED (
    SELECT id
    FROM prediction_jobs
    WHERE status = 'queued'
        AND attempt_count >= sqlc.arg(max_attempts)::integer
        AND sqlc.arg(max_attempts)::integer > 0
    ORDER BY created_at, id
    LIMIT sqlc.arg(batch_size)::integer
    FOR UPDATE SKIP LOCKED
)
UPDATE prediction_jobs AS job
SET status = 'failed',
    finished_at = clock_timestamp(),
    lease_until = NULL,
    last_error_code = 'attempt_limit_exceeded',
    last_error_message = 'The configured attempt limit has been reached'
FROM exhausted_jobs
WHERE job.id = exhausted_jobs.id
RETURNING job.id, job.status, job.attempt_count;
