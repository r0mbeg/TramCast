"""Run the serving and client contract checks, without research dependencies."""
import os
from tests import test_service, test_client, test_cache, test_recipe, test_inference_service, test_stops

test_service.check()
test_service.check(test_service.ROOT.parent / "bundles" / "030" / "forecast_bundle.json")
test_client.check()

test_cache.check()
test_recipe.check()
test_inference_service.check()
test_stops.check(os.environ.get('STOP_REFERENCE_FILE'))
