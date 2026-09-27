"""Fail startup if the deployment's essential sandbox restrictions are absent."""
import errno
import os
from pathlib import Path
import socket
import tempfile

home = Path.home()
root = home / "tramcast"
status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
assert status["NoNewPrivs"].strip() == "1", "Privilege escalation is not blocked"
assert int(status["CapEff"].strip(), 16) == 0, "Unexpected effective capabilities"
assert not (home / ".ssh").exists(), "SSH directory is visible inside the service"
assert not os.access(f"/run/user/{os.getuid()}", os.R_OK | os.X_OK), "User session sockets are accessible"
for process in Path("/proc").iterdir():
    if not process.name.isdecimal():
        continue
    try:
        assert not (process / "root" / home.relative_to("/") / ".ssh").exists(), "SSH directory is reachable through /proc"
    except (PermissionError, ProcessLookupError, FileNotFoundError):
        pass
for family in (socket.AF_INET, socket.AF_INET6):
    try:
        with socket.socket(family, socket.SOCK_STREAM):
            raise AssertionError("Internet sockets are not blocked")
    except OSError as error:
        assert error.errno in (errno.EAFNOSUPPORT, errno.EPERM, errno.EACCES), error
try:
    with tempfile.TemporaryFile(dir=root):
        raise AssertionError("Application directory is writable")
except OSError as error:
    assert error.errno in (errno.EROFS, errno.EACCES), error
for directory in (root / "run", root / "ml/cache"):
    assert directory.stat().st_mode & 0o077 == 0, f"Public directory: {directory}"
    with tempfile.TemporaryFile(dir=directory):
        pass
cgroup = Path("/sys/fs/cgroup") / Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1].lstrip("/")
assert int((cgroup / "memory.max").read_text()) <= 8 * 1024**3, "Memory limit is absent"
assert len(os.sched_getaffinity(0)) <= 2, "CPU affinity is not restricted"
assert int((cgroup / "pids.max").read_text()) <= 128, "Task limit is absent"
try:
    with (cgroup / "memory.max").open("w"):
        raise AssertionError("Service can change its own resource limits")
except OSError as error:
    assert error.errno in (errno.EROFS, errno.EACCES), error
print("Sandbox verified: no IP sockets/capabilities/SSH access; read-only code; private writable state; resource limits.", flush=True)
