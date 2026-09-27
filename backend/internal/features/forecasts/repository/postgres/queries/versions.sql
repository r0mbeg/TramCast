-- name: CreateForecastVersion :one
-- Versions are registered inactive. Metadata is immutable after creation.
-- When the same artifacts and period are registered, no row is returned:
-- issue GetForecastVersionByArtifacts as a NEW statement (fresh snapshot).
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
ON CONFLICT ON CONSTRAINT forecast_versions_artifacts_key DO NOTHING
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

-- name: GetForecastVersionByArtifacts :one
-- Matches forecast_versions_artifacts_key; timestamps compare as instants.
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
WHERE model_version = sqlc.arg(model_version)::text
    AND dataset_version = sqlc.arg(dataset_version)::text
    AND history_end = sqlc.arg(history_end)::timestamptz
    AND forecast_from = sqlc.arg(forecast_from)::timestamptz
    AND forecast_to = sqlc.arg(forecast_to)::timestamptz;

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
