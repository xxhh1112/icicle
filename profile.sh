#!/bin/bash

rep_name=${1:-"run-$2-$3"}
# cargo +nightly test --release --package icicle-utils --lib -- test_bls12_381::tests_bls12_381::test_scalar_batch_fast_ntt --exact --nocapture; 
# RUSTFLAGS=-Awarnings cargo +nightly run --release; /tmp/var/target/linux-desktop-glibc_2_11_3-x64/ncu --config-file off --export /tmp/var/$rep_name --force-overwrite --kernel-id ::regex:'^(?!twiddle)': --section-folder /tmp/var/sections --set full --sampling-max-passes 1 /home/vlad/Projects/icicle_clean/icicle/target/release/icicle-utils

# capture the output
output=$(RUSTFLAGS=-Awarnings cargo +nightly run --release)

# Now we parse the output using a regex to find the time value
# The regex here looks for the word "batch", followed by "1024", a space, and a decimal number
regex='batch 1024 ([0-9]*\.[0-9]*)'
if [[ $output =~ $regex ]]; then
    # If the regex matches, the time value will be in the bash variable BASH_REMATCH[1]
    time=${BASH_REMATCH[1]}
    
    # Print the time value
    echo "batch 1024*1024: $time us"
    
    # If the time is greater than 1000, we print a warning
    if (( $(echo "$time > 2800" | bc -l) )); then
        echo "Warning: High time value!"
        # You can launch another command here
        # For example, echo "Launching another command..."
        # ./another-command.sh
    else
        echo "profiling..."
        /tmp/var/target/linux-desktop-glibc_2_11_3-x64/ncu --config-file off --export /tmp/var/run-ingo-ntt+ari_bail_operations-limit_r40-byblock-a3b2c93ff-2 --force-overwrite --kernel-id ::regex:'^(?!twiddle)': --launch-count 1 --section-folder /tmp/var/sections --set full --sampling-max-passes 1 /home/vlad/Projects/icicle_clean/icicle/target/release/icicle-utils
    fi
else
    echo "No time value found in output."
fi

