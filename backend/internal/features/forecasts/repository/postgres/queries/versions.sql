-- name: CreateForecastVersion :one
-- Versions are registered inactive. Metadata is immutable after creation.
INSERT INTO forecast_versions (
    id,
    model_version,
    dataset_version,
    history_end,
    forecast_from,
    forecast_to
) VALUES (
    sqlc.arg(id)::uuid,
    sqlc.arg(model_version)::text,
    sqlc.arg(dataset_version)::text,
    sqlc.arg(history_end)::timestamptz,
    sqlc.arg(forecast_from)::timestamptz,
    sqlc.arg(forecast_to)::timestamptz
)
RETURNING
    id,
    model_version,
    dataset_version,
    history_end,
    forecast_from,
    forecast_to,
    timezone,
    is_active,
    created_at;

-- name: GetForecastVersion :one
SELECT
    id,
    model_version,
    dataset_version,
    history_end,
    forecast_from,
    forecast_to,
    timezone,
    is_active,
    created_at
FROM forecast_versions
WHERE id = sqlc.arg(id)::uuid;

-- name: GetActiveForecastVersion :one
SELECT
    id,
    model_version,
    dataset_version,
    history_end,
    forecast_from,
    forecast_to,
    timezone,
    is_active,
    created_at
FROM forecast_versions
WHERE is_active = true;

-- name: LockForecastVersionsForActivation :exec
-- Use a short READ COMMITTED transaction: lock, GetForecastVersion (rollback if
-- missing), DeactivateForecastVersions, ActivateForecastVersion, commit.
-- The table lock serializes switches even when there is no active version yet.
-- All activation queries must use the same transaction; readers are not blocked.
LOCK TABLE forecast_versions IN SHARE ROW EXCLUSIVE MODE;

-- name: DeactivateForecastVersions :exec
-- Call only after LockForecastVersionsForActivation in the switch transaction.
UPDATE forecast_versions
SET is_active = false
WHERE is_active = true;

-- name: ActivateForecastVersion :one
-- Call after deactivation in the same transaction. Roll back on any error.
UPDATE forecast_versions
SET is_active = true
WHERE id = sqlc.arg(id)::uuid
RETURNING
    id,
    model_version,
    dataset_version,
    history_end,
    forecast_from,
    forecast_to,
    timezone,
    is_active,
    created_at;
