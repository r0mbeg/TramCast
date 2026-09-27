-- name: ListRoutePatterns :many
SELECT pattern_key, direction_id
FROM routes_stops
WHERE route_id = sqlc.arg(route_id)
GROUP BY pattern_key, direction_id
ORDER BY direction_id, pattern_key;

-- name: ListRouteStops :many
SELECT rs.route_id,
       rs.pattern_key,
       rs.direction_id,
       rs.stop_sequence,
       s.id AS stop_id,
       s.source_stop_id,
       s.name,
       s.latitude,
       s.longitude
FROM routes_stops AS rs
JOIN stops AS s ON s.id = rs.stop_id
WHERE rs.route_id = sqlc.arg(route_id)
ORDER BY rs.direction_id, rs.pattern_key, rs.stop_sequence;

-- name: ListRouteGeometry :many
-- All positions of all routes with stop coordinates, ordered so that the rows
-- of one movement variant are adjacent and follow stop_sequence.
SELECT rs.route_id,
       r.route_number,
       r.forecast_enabled,
       rs.pattern_key,
       rs.direction_id,
       rs.stop_sequence,
       s.latitude,
       s.longitude
FROM routes_stops AS rs
JOIN routes AS r ON r.id = rs.route_id
JOIN stops AS s ON s.id = rs.stop_id
ORDER BY r.route_number, rs.route_id, rs.direction_id, rs.pattern_key, rs.stop_sequence;

-- name: DeleteRouteStops :execrows
-- Use only within the replacement transaction after GetRouteForUpdate.
DELETE FROM routes_stops
WHERE route_id = sqlc.arg(route_id);

-- name: InsertRouteStop :exec
-- In the same transaction, insert the validated full set after DeleteRouteStops.
-- Go verifies one direction_id per pattern_key before replacing any positions.
INSERT INTO routes_stops (route_id, pattern_key, direction_id, stop_sequence, stop_id)
VALUES (
    sqlc.arg(route_id),
    sqlc.arg(pattern_key),
    sqlc.arg(direction_id),
    sqlc.arg(stop_sequence),
    sqlc.arg(stop_id)
);
