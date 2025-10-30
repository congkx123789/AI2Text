set -euo pipefail
find data/interim/uploads -type f -mtime +7 -delete || true