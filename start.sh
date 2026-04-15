#!/usr/bin/env bash
set -euo pipefail

mkdir -p results

SCANNER_MODE="${SCANNER_MODE:-nps}"
WORKERS="${WORKERS:-5}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
TIMESTAMP="$(date +"%Y%m%d_%H%M")"
EXTRA_TICKERS="${TICKERS:-}"
NO_DETAIL_FLAG="${NO_DETAIL:-0}"
QUIET_FLAG="${QUIET:-0}"

mkdir -p "$OUTPUT_DIR"

build_common_args() {
  local args=(--workers "$WORKERS" --log-level "$LOG_LEVEL")
  if [[ "$QUIET_FLAG" == "1" ]]; then
    args+=(--quiet)
  fi
  if [[ -n "$EXTRA_TICKERS" ]]; then
    # shellcheck disable=SC2206
    local tickers_array=( $EXTRA_TICKERS )
    args+=(--tickers "${tickers_array[@]}")
  else
    args+=(--file ticker.txt)
  fi
  printf '%s\n' "${args[@]}"
}

mapfile -t COMMON_ARGS < <(build_common_args)

if [[ "$SCANNER_MODE" == "hgv" ]]; then
  OUTPUT_FILE="$OUTPUT_DIR/hgv_${TIMESTAMP}.csv"
  CMD=(python high_growth_valuation.py "${COMMON_ARGS[@]}" --output "$OUTPUT_FILE")
  if [[ "$NO_DETAIL_FLAG" == "1" ]]; then
    CMD+=(--no-score-detail)
  fi
  echo "[Railway] Running High Growth Valuation scanner..."
else
  OUTPUT_FILE="$OUTPUT_DIR/nps_${TIMESTAMP}.csv"
  CMD=(python next_palantir_scanner.py "${COMMON_ARGS[@]}" --output "$OUTPUT_FILE")
  if [[ "$NO_DETAIL_FLAG" == "1" ]]; then
    CMD+=(--no-detail)
  fi
  echo "[Railway] Running Next Palantir scanner..."
fi

printf '[Railway] Command: '
printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

if [[ -f "$OUTPUT_FILE" ]]; then
  cp "$OUTPUT_FILE" "$OUTPUT_DIR/latest.csv"
  echo "[Railway] Latest output: $OUTPUT_DIR/latest.csv"
fi
