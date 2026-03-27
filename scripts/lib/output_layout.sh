#!/usr/bin/env bash

dz_output_timestamp_date() {
    if [ -n "${DREAMZERO_OUTPUT_DATE:-}" ]; then
        printf '%s\n' "$DREAMZERO_OUTPUT_DATE"
    else
        date +"%Y-%m-%d"
    fi
}

dz_output_timestamp_time() {
    if [ -n "${DREAMZERO_OUTPUT_TIME:-}" ]; then
        printf '%s\n' "$DREAMZERO_OUTPUT_TIME"
    else
        date +"%H-%M-%S"
    fi
}

dz_default_output_root() {
    local repo_root="$1"
    printf '%s\n' "${DREAMZERO_OUTPUT_ROOT:-$repo_root/outputs}"
}

dz_default_train_dir() {
    local output_root="$1"
    local embodiment="$2"
    local train_mode="$3"
    local run_date run_time
    run_date="$(dz_output_timestamp_date)"
    run_time="$(dz_output_timestamp_time)"
    printf '%s\n' "$output_root/train/$run_date/${run_time}-${embodiment}-${train_mode}"
}

dz_default_inference_dir() {
    local output_root="$1"
    local checkpoint_name="$2"
    local run_date run_time
    run_date="$(dz_output_timestamp_date)"
    run_time="$(dz_output_timestamp_time)"
    printf '%s\n' "$output_root/inference/$run_date/${run_time}-${checkpoint_name}"
}
