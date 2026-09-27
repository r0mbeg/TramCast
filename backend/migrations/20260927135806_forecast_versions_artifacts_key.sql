-- +goose Up
-- One version per set of artifacts and period: registering it again finds the
-- existing row instead of splitting jobs and predictions between duplicates.
ALTER TABLE forecast_versions
    ADD CONSTRAINT forecast_versions_artifacts_key
    UNIQUE (model_version, dataset_version, history_end, forecast_from, forecast_to);

-- +goose Down
ALTER TABLE forecast_versions DROP CONSTRAINT forecast_versions_artifacts_key;
