"""Run the serving and client contract checks, without research dependencies."""
from tests import test_service, test_client, test_cache, test_recipe, test_inference_service

test_service.check()
test_service.check(test_service.ROOT.parent / "bundles" / "030" / "forecast_bundle.json")
test_client.check()

test_cache.check()
test_recipe.check()
test_inference_service.check()
