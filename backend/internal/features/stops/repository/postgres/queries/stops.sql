-- name: GetStopByID :one
SELECT id, source_stop_id, name, latitude, longitude
FROM stops
WHERE id = sqlc.arg(stop_id);

-- name: GetStopBySourceID :one
SELECT id, source_stop_id, name, latitude, longitude
FROM stops
WHERE source_stop_id = sqlc.arg(source_stop_id);

-- name: UpsertStop :one
INSERT INTO stops (source_stop_id, name, latitude, longitude)
VALUES (
    sqlc.arg(source_stop_id),
    sqlc.arg(name),
    sqlc.arg(latitude),
    sqlc.arg(longitude)
)
ON CONFLICT (source_stop_id) DO UPDATE
SET name = EXCLUDED.name,
    latitude = EXCLUDED.latitude,
    longitude = EXCLUDED.longitude
RETURNING id, source_stop_id, name, latitude, longitude;
