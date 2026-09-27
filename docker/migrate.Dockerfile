# syntax=docker/dockerfile:1

# BuildKit provides TARGETARCH in the global scope used by FROM. Do not
# redeclare it here: older bundled frontends can replace its value with empty.
FROM debian:bookworm AS goose-amd64
ADD --chmod=0755 --checksum=sha256:ab073515b78ef345f64f018c0d79aa7db50106806efc686dda7181253765ae13 https://github.com/pressly/goose/releases/download/v3.28.0/goose_linux_x86_64 /usr/local/bin/goose

FROM debian:bookworm AS goose-arm64
ADD --chmod=0755 --checksum=sha256:3968855c11b4093af271c5226909789ea294468f6e50203d738b7504995b6247 https://github.com/pressly/goose/releases/download/v3.28.0/goose_linux_arm64 /usr/local/bin/goose

FROM goose-${TARGETARCH} AS final
ENTRYPOINT ["/usr/local/bin/goose"]
