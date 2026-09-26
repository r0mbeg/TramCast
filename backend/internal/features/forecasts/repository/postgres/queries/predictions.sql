-- name: ListValidationPredictions :many
-- Go converts validated, hour-aligned bounds to Europe/Moscow date/hour pairs.
-- The interval is [from, to); verify the exact hourly grid, not just its size.
SELECT
    forecast_version_id,
    route_id,
    date,
    hour,
    boardings,
    created_at
FROM validation_predictions
WHERE forecast_version_id = sqlc.arg(forecast_version_id)::uuid
    AND route_id = sqlc.arg(route_id)::bigint
    AND (date, hour) >= (sqlc.arg(date_from)::date, sqlc.arg(hour_from)::smallint)
    AND (date, hour) < (sqlc.arg(date_to)::date, sqlc.arg(hour_to)::smallint)
ORDER BY date, hour;

-- name: CopyValidationPredictions :copyfrom
-- Copy the verified full route/version horizon using Queries.WithTx(tx).
-- First lock the owned running job and check attempt_count and its valid lease;
-- then copy every point and mark success in this same transaction. Any error,
-- missing ownership, or an unexpected copy count requires rollback of the batch.
-- Validate point metadata and the exact grid before the transaction. Published
-- values are immutable: there is deliberately no UPDATE or ON CONFLICT clause.
INSERT INTO validation_predictions (
    forecast_version_id,
    route_id,
    date,
    hour,
    boardings
) VALUES (
    sqlc.arg(forecast_version_id),
    sqlc.arg(route_id),
    sqlc.arg(date),
    sqlc.arg(hour),
    sqlc.arg(boardings)
);
