#!/usr/bin/env bash
# Measures a running TramCast from the server itself, so network latency to
# users is left out: HTTP load with ab, resources with docker stats and vmstat,
# and a cold CPU forecast into a temporary ML cache (the real cache and the
# database are not touched). Run from any directory after make app-up.
#
#   bash scripts/perf.sh [base URL]      default http://127.0.0.1:<HTTP_PORT of .env>
#
# Needs curl, Docker Compose and ab (apt install apache2-utils).
# PERF_DURATION sets the seconds of each load step (default 30).
# Raw output goes to out/perf/<UTC time>/; the summary is printed as Markdown.
set -euo pipefail
cd "$(dirname "$0")/.."

for tool in ab curl docker; do
  command -v "$tool" >/dev/null || { echo "$tool is required (ab: apt install apache2-utils)" >&2; exit 1; }
done

port=8080
if [ -f .env ]; then
  value=$(sed -n 's/^HTTP_PORT=//p' .env | tail -n 1 | tr -d "'\"[:space:]")
  port=${value:-$port}
fi
base=${1:-http://127.0.0.1:$port}
duration=${PERF_DURATION:-30}
out=out/perf/$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$out"

curl -fsS "$base/readyz" >/dev/null || { echo "$base/readyz is not ready; run make app-up first" >&2; exit 1; }
docker compose ps --status running --services | grep -qx ml || { echo "the ml service is not running; run make app-up first" >&2; exit 1; }
version=$(curl -fsS "$base/api/forecast-versions/active")
version_id=$(sed -E 's/.*"id":"([^"]+)".*/\1/' <<<"$version")
model_version=$(sed -E 's/.*"model_version":"([^"]+)".*/\1/' <<<"$version")
# The internal ID of route 1; HTTP takes internal IDs, not route numbers.
route_id=$(curl -fsS "$base/api/routes" | grep -oE '"id":[0-9]+,"route_number":1,' | grep -oE '[0-9]+' | head -n 1)
[ -n "$route_id" ] || { echo "route 1 is not in the catalog" >&2; exit 1; }

day_url="$base/api/predictions?forecast_version_id=$version_id&route_id=$route_id&from=2025-11-05T00%3A00%3A00%2B03%3A00&to=2025-11-06T00%3A00%3A00%2B03%3A00"
routes_url="$base/api/routes"
query_url="$base/api/predictions/query"
printf '{"route_id":%s,"from":"2025-11-01T00:00:00+03:00","to":"2026-01-01T00:00:00+03:00"}' "$route_id" >"$out/query.json"

{
  echo "Дата (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Коммит: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)$( [ -n "$(git status --porcelain 2>/dev/null)" ] && echo ' (есть незакоммиченные изменения)')"
  echo "Модель: $model_version"
  echo "Сервер: $(nproc) CPU, $(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -n 1), RAM $(free -h | awk '/^Mem:/ {print $2}'), swap $(free -h | awk '/^Swap:/ {print $2}')"
  echo "ОС: $(. /etc/os-release && echo "$PRETTY_NAME"), ядро $(uname -r)"
  echo "Docker $(docker version --format '{{.Server.Version}}'), Compose $(docker compose version --short)"
  echo "Лимиты контейнеров (0 — нет): $(docker inspect -f '{{.Name}} mem={{.HostConfig.Memory}} cpus={{.HostConfig.NanoCpus}}' $(docker compose ps -q) | tr '\n' ' ')"
  echo "Нагрузка: ApacheBench (ab, HTTP/1.0) на том же сервере, адрес $base, ${duration} с на ступень"
} | tee "$out/conditions.txt"

# Peak CPU % and memory (MiB) per container from docker stats samples.
stats_peaks() {
  awk -F, '
    function mib(v,  n) {
      n = v + 0
      if (v ~ /GiB/) return n * 1024
      if (v ~ /MiB/) return n
      if (v ~ /KiB|kB/) return n / 1024
      return n / 1048576
    }
    {
      name = $2; sub(/^tramcast-/, "", name); sub(/-1$/, "", name)
      cpu = $3; sub(/%/, "", cpu)
      split($4, mem, " / ")
      if (cpu + 0 > peak_cpu[name]) peak_cpu[name] = cpu + 0
      if (mib(mem[1]) > peak_mem[name]) peak_mem[name] = mib(mem[1])
    }
    END {
      # docker stats takes about 2 s per sample: a shorter step may have none.
      if (NR == 0) { printf "— / — / —|— / — / —"; exit }
      printf "%.0f / %.0f / %.0f|%.0f / %.0f / %.0f", peak_cpu["backend"], peak_cpu["ml"], peak_cpu["postgres"], peak_mem["backend"], peak_mem["ml"], peak_mem["postgres"]
    }' "$1"
}

# Busiest whole-server CPU % and swap activity from vmstat; the first line
# averages since boot and is skipped.
vmstat_peaks() {
  awk 'NR > 3 && $1 ~ /^[0-9]+$/ { n++; busy = 100 - $15; if (busy > max) max = busy; swap += $7 + $8; if ($3 > used) used = $3 }
       END { if (n == 0) { printf "—|—"; exit } printf "%d|%d КиБ, si+so %d", max, used, swap }' "$1"
}

summary=$out/summary.md
{
  echo "| Сценарий | Клиенты | Запросов | RPS | p50, мс | p95, мс | p99, мс | Ошибки | CPU backend / ML / PG, % | Память backend / ML / PG, МиБ | CPU сервера, % | Swap |"
  echo "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|"
} >"$summary"

step() {
  local name=$1 clients=$2 url=$3
  shift 3
  local tag
  tag=$(tr -c 'a-z0-9' '_' <<<"$name-$clients" | tr -s '_')
  echo "== $name, $clients клиентов, $duration с" >&2
  : >"$out/$tag.stats"
  (while :; do docker stats --no-stream --format '{{.Name}},{{.CPUPerc}},{{.MemUsage}}' | sed "s/^/$(date +%s),/" >>"$out/$tag.stats"; done) &
  local sampler=$!
  vmstat -n 2 >"$out/$tag.vmstat" &
  local vm=$!
  # -t before -n: ab sets 50 000 requests for -t, the larger -n lifts it.
  ab -k -l -r -q -c "$clients" -t "$duration" -n 100000000 "$@" "$url" >"$out/$tag.ab" 2>&1 || true
  kill "$sampler" "$vm" 2>/dev/null || true
  wait "$sampler" "$vm" 2>/dev/null || true
  local ab=$out/$tag.ab
  local complete rps p50 p95 p99 failed non2xx
  complete=$(awk '/^Complete requests:/ {print $3}' "$ab")
  rps=$(awk '/^Requests per second:/ {printf "%.0f", $4}' "$ab")
  p50=$(awk '$1 == "50%" {print $2}' "$ab")
  p95=$(awk '$1 == "95%" {print $2}' "$ab")
  p99=$(awk '$1 == "99%" {print $2}' "$ab")
  failed=$(awk '/^Failed requests:/ {print $3}' "$ab")
  non2xx=$(awk '/^Non-2xx responses:/ {print $3}' "$ab")
  local peaks vm_peaks
  peaks=$(stats_peaks "$out/$tag.stats")
  vm_peaks=$(vmstat_peaks "$out/$tag.vmstat")
  echo "| $name | $clients | ${complete:-0} | ${rps:-0} | ${p50:--} | ${p95:--} | ${p99:--} | $(( ${failed:-0} + ${non2xx:-0} )) | ${peaks%%|*} | ${peaks#*|} | ${vm_peaks%%|*} | ${vm_peaks#*|} |" >>"$summary"
  sleep 5
}

for clients in 10 50 100 200; do
  step "Сутки маршрута, GET /api/predictions" "$clients" "$day_url"
done
step "Справочник, GET /api/routes" 50 "$routes_url"
for clients in 10 50; do
  step "Весь горизонт, POST /api/predictions/query" "$clients" "$query_url" -p "$out/query.json" -T application/json
done

# Cold forecast: the CPU model computes all routes over the whole horizon into
# a new temporary cache, as for a new version; the time includes docker exec.
echo "== Холодный расчёт полного горизонта" >&2
docker compose exec -T ml rm -f /tmp/perf-cold.sqlite3
start=$(date +%s.%N)
docker compose exec -T ml python service.py --warm-cache --cache /tmp/perf-cold.sqlite3 >"$out/cold.txt" 2>&1
end=$(date +%s.%N)
docker compose exec -T ml rm -f /tmp/perf-cold.sqlite3
points=$(grep -oE '"points": ?[0-9]+' "$out/cold.txt" | grep -oE '[0-9]+$' || echo "?")
cold=$(awk -v a="$start" -v b="$end" 'BEGIN {printf "%.1f", b - a}')

{
  echo
  echo "Холодный расчёт CPU-моделью: $cold с, $points часов (10 маршрутов × 1 464)."
} >>"$summary"

echo
cat "$out/conditions.txt"
echo
cat "$summary"
echo
echo "Сырые данные: $out"
