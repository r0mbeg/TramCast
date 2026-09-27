import hashlib, json, resource, time
from importlib.metadata import version
from pathlib import Path
import torch
import model_030
assert torch.cuda.is_available()
total = torch.cuda.get_device_properties(0).total_memory
budget = 6 * 1024**3
torch.cuda.set_per_process_memory_fraction(min(1., budget / total))
torch.cuda.reset_peak_memory_stats()
started = time.monotonic()
model_030.run("spec.json", "forecast.csv")
torch.cuda.synchronize()
report = dict(seconds=time.monotonic()-started, gpu=torch.cuda.get_device_name(0),
    total_bytes=total, allocator_limit_bytes=budget,
    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    sha256=hashlib.sha256(Path("forecast.csv").read_bytes()).hexdigest(),
    versions={p:version(p) for p in ("torch","tabpfn","numpy","pandas","scikit-learn","scipy")})
Path("verification.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2),flush=True)
