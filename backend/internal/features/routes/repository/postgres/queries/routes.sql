-- name: ListRoutes :many
SELECT id, route_number, name, source_route_id, forecast_enabled
FROM routes
ORDER BY route_number, id;

-- name: GetRouteByID :one
SELECT id, route_number, name, source_route_id, forecast_enabled
FROM routes
WHERE id = sqlc.arg(route_id);

-- name: GetRouteByNumber :one
SELECT id, route_number, name, source_route_id, forecast_enabled
FROM routes
WHERE route_number = sqlc.arg(route_number);

-- name: GetRouteBySourceID :one
SELECT id, route_number, name, source_route_id, forecast_enabled
FROM routes
WHERE source_route_id = sqlc.arg(source_route_id)::text;

-- name: EnsureRoute :execrows
-- After this statement, read the existing/new row with GetRouteByNumber.
INSERT INTO routes (route_number)
VALUES (sqlc.arg(route_number))
ON CONFLICT (route_number) DO NOTHING;

-- name: UpsertRoute :one
-- Missing catalog fields preserve existing values; enabling forecasts is separate.
INSERT INTO routes (route_number, name, source_route_id)
VALUES (
    sqlc.arg(route_number),
    sqlc.narg(name)::text,
    sqlc.narg(source_route_id)::text
)
ON CONFLICT (route_number) DO UPDATE
SET name = COALESCE(EXCLUDED.name, routes.name),
    source_route_id = COALESCE(EXCLUDED.source_route_id, routes.source_route_id)
RETURNING id, route_number, name, source_route_id, forecast_enabled;

-- name: SetRouteForecastEnabled :execrows
UPDATE routes
SET forecast_enabled = sqlc.arg(forecast_enabled)
WHERE id = sqlc.arg(route_id);

-- name: GetRouteForUpdate :one
-- Lock before replacing this route's positions; hold through delete and inserts.
SELECT id, route_number, name, source_route_id, forecast_enabled
FROM routes
WHERE id = sqlc.arg(route_id)
FOR UPDATE;
