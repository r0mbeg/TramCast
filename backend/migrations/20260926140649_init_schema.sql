-- +goose Up
CREATE TABLE routes (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    route_number smallint NOT NULL,
    name text,
    source_route_id text,
    forecast_enabled boolean NOT NULL DEFAULT false,

    CONSTRAINT routes_route_number_key UNIQUE (route_number),
    CONSTRAINT routes_source_route_id_key UNIQUE (source_route_id),
    CONSTRAINT routes_route_number_positive CHECK (route_number > 0),
    CONSTRAINT routes_source_route_id_not_blank CHECK (
        source_route_id IS NULL OR btrim(source_route_id) <> ''
    )
);

CREATE TABLE stops (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_stop_id text NOT NULL,
    name text NOT NULL,
    latitude double precision NOT NULL,
    longitude double precision NOT NULL,

    CONSTRAINT stops_source_stop_id_key UNIQUE (source_stop_id),
    CONSTRAINT stops_source_stop_id_not_blank CHECK (btrim(source_stop_id) <> ''),
    CONSTRAINT stops_name_not_blank CHECK (btrim(name) <> ''),
    -- These ranges also reject NaN and positive/negative infinity.
    CONSTRAINT stops_latitude_range CHECK (latitude BETWEEN -90 AND 90),
    CONSTRAINT stops_longitude_range CHECK (longitude BETWEEN -180 AND 180)
);

CREATE TABLE routes_stops (
    route_id bigint NOT NULL,
    pattern_key text NOT NULL,
    direction_id smallint NOT NULL,
    stop_sequence integer NOT NULL,
    stop_id bigint NOT NULL,

    PRIMARY KEY (route_id, pattern_key, stop_sequence),
    CONSTRAINT routes_stops_route_id_fkey FOREIGN KEY (route_id)
        REFERENCES routes (id) ON DELETE RESTRICT,
    CONSTRAINT routes_stops_stop_id_fkey FOREIGN KEY (stop_id)
        REFERENCES stops (id) ON DELETE RESTRICT,
    CONSTRAINT routes_stops_pattern_key_not_blank CHECK (btrim(pattern_key) <> ''),
    CONSTRAINT routes_stops_direction_id_valid CHECK (direction_id IN (0, 1)),
    CONSTRAINT routes_stops_stop_sequence_nonnegative CHECK (stop_sequence >= 0)
);

CREATE INDEX routes_stops_stop_id_idx ON routes_stops (stop_id);

CREATE TABLE forecast_versions (
    id uuid PRIMARY KEY,
    model_version text NOT NULL,
    dataset_version text NOT NULL,
    history_end timestamptz NOT NULL,
    forecast_from timestamptz NOT NULL,
    forecast_to timestamptz NOT NULL,
    timezone text NOT NULL DEFAULT 'Europe/Moscow',
    is_active boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT forecast_versions_model_version_not_blank CHECK (btrim(model_version) <> ''),
    CONSTRAINT forecast_versions_dataset_version_not_blank CHECK (btrim(dataset_version) <> ''),
    CONSTRAINT forecast_versions_boundaries_finite CHECK (
        isfinite(history_end) AND isfinite(forecast_from) AND isfinite(forecast_to)
    ),
    CONSTRAINT forecast_versions_period_valid CHECK (
        history_end <= forecast_from AND forecast_from < forecast_to
    ),
    CONSTRAINT forecast_versions_timezone_valid CHECK (timezone = 'Europe/Moscow'),
    -- Compare local timestamps explicitly, independently of the session TimeZone.
    CONSTRAINT forecast_versions_from_hour_aligned CHECK (
        forecast_from AT TIME ZONE 'Europe/Moscow'
        = date_trunc('hour', forecast_from AT TIME ZONE 'Europe/Moscow')
    ),
    CONSTRAINT forecast_versions_to_hour_aligned CHECK (
        forecast_to AT TIME ZONE 'Europe/Moscow'
        = date_trunc('hour', forecast_to AT TIME ZONE 'Europe/Moscow')
    )
);

CREATE UNIQUE INDEX forecast_versions_one_active_idx
    ON forecast_versions (is_active)
    WHERE is_active = true;

CREATE TABLE validation_predictions (
    forecast_version_id uuid NOT NULL,
    route_id bigint NOT NULL,
    date date NOT NULL,
    hour smallint NOT NULL,
    boardings bigint NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),

    PRIMARY KEY (forecast_version_id, route_id, date, hour),
    CONSTRAINT validation_predictions_forecast_version_id_fkey FOREIGN KEY (forecast_version_id)
        REFERENCES forecast_versions (id) ON DELETE RESTRICT,
    CONSTRAINT validation_predictions_route_id_fkey FOREIGN KEY (route_id)
        REFERENCES routes (id) ON DELETE RESTRICT,
    CONSTRAINT validation_predictions_date_finite CHECK (isfinite(date)),
    CONSTRAINT validation_predictions_hour_valid CHECK (hour BETWEEN 0 AND 23),
    CONSTRAINT validation_predictions_boardings_nonnegative CHECK (boardings >= 0)
);

CREATE INDEX validation_predictions_route_id_idx ON validation_predictions (route_id);

CREATE TABLE prediction_jobs (
    id uuid PRIMARY KEY,
    forecast_version_id uuid NOT NULL,
    route_id bigint NOT NULL,
    status text NOT NULL DEFAULT 'queued',
    attempt_count integer NOT NULL DEFAULT 0,
    run_after timestamptz NOT NULL DEFAULT now(),
    lease_until timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    finished_at timestamptz,
    last_error_code text,
    last_error_message text,

    -- Deduplicate every status, including completed and failed jobs.
    CONSTRAINT prediction_jobs_forecast_version_route_key UNIQUE (forecast_version_id, route_id),
    CONSTRAINT prediction_jobs_forecast_version_id_fkey FOREIGN KEY (forecast_version_id)
        REFERENCES forecast_versions (id) ON DELETE RESTRICT,
    CONSTRAINT prediction_jobs_route_id_fkey FOREIGN KEY (route_id)
        REFERENCES routes (id) ON DELETE RESTRICT,
    CONSTRAINT prediction_jobs_status_valid CHECK (
        status IN ('queued', 'running', 'succeeded', 'failed')
    ),
    CONSTRAINT prediction_jobs_attempt_count_nonnegative CHECK (attempt_count >= 0),
    -- Lease expiry and ownership are checked by conditional worker queries.
    CONSTRAINT prediction_jobs_state_valid CHECK (
        (status = 'queued' AND lease_until IS NULL AND finished_at IS NULL)
        OR (
            status = 'running'
            AND attempt_count > 0
            AND started_at IS NOT NULL
            AND lease_until IS NOT NULL
            AND finished_at IS NULL
        )
        OR (status IN ('succeeded', 'failed') AND finished_at IS NOT NULL AND lease_until IS NULL)
    )
);

CREATE INDEX prediction_jobs_route_id_idx ON prediction_jobs (route_id);

CREATE INDEX prediction_jobs_queued_idx
    ON prediction_jobs (run_after, created_at)
    WHERE status = 'queued';

CREATE INDEX prediction_jobs_running_lease_idx
    ON prediction_jobs (lease_until)
    WHERE status = 'running';

-- +goose Down
DROP TABLE prediction_jobs;
DROP TABLE validation_predictions;
DROP TABLE forecast_versions;
DROP TABLE routes_stops;
DROP TABLE stops;
DROP TABLE routes;
