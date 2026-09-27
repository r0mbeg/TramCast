-- name: LockCatalogImport :exec
-- Serializes catalog imports; the lock is released at commit or rollback.
SELECT pg_advisory_xact_lock(hashtext('tramcast.catalog_import'));

-- name: DeleteUnusedStopsNotInCatalog :execrows
-- Run after positions are replaced: removes stops absent from the imported
-- sources that no position references any more.
DELETE FROM stops AS s
WHERE NOT (s.source_stop_id = ANY (sqlc.arg(source_stop_ids)::text[]))
  AND NOT EXISTS (
      SELECT 1
      FROM routes_stops AS rs
      WHERE rs.stop_id = s.id
  );
