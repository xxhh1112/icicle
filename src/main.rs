use  nvtx::*;

use icicle_utils::{
    test_bls12_381::{
        evaluate_points_batch_bls12_381, evaluate_scalars_batch_bls12_381,
        interpolate_points_batch_bls12_381, interpolate_scalars_batch_bls12_381,
        ntt_batch_bls12_381, set_up_points_bls12_381, set_up_scalars_bls12_381, get_rng_bls12_381, generate_random_scalars_bls12_381, intt_batch_bls12_381,
    },
};
use rustacuda::prelude::DeviceBuffer;

const LOG_NTT_SIZES: [usize; 1] = [10]; //, 23, 9, 10, 11, 12, 18];
const BATCH_SIZES: [usize; 1] = [1<<10]; //, 4, 8, 16, 256, 512, 1024, 1 << 14];
// const LOG_NTT_SIZES: [usize; 4] = [10, 12, 23, 24]; //, 23, 9, 10, 11, 12, 18];
// const BATCH_SIZES: [usize; 6] = [1, 2, 4, 16, 256, 1<<16]; //, 4, 8, 16, 256, 512, 1024, 1 << 14];

const MAX_POINTS_LOG2: usize = 10;
const MAX_SCALARS_LOG2: usize = 26;


fn bench_lde() {
    for log_ntt_size in LOG_NTT_SIZES {
        for batch_size in BATCH_SIZES {
            let ntt_size = 1 << log_ntt_size;         

            bench_template(
                MAX_SCALARS_LOG2,
                ntt_size,
                batch_size,
                log_ntt_size,
                set_up_scalars_bls12_381,
                evaluate_scalars_batch_bls12_381,
                "NTT",
                false,
                10,
            );
            bench_template(
                MAX_SCALARS_LOG2,
                ntt_size,
                batch_size,
                log_ntt_size,
                set_up_scalars_bls12_381,
                interpolate_scalars_batch_bls12_381,
                "iNTT",
                true,
                10,
            );
            bench_template(
                MAX_POINTS_LOG2,
                ntt_size,
                batch_size,
                log_ntt_size,
                set_up_points_bls12_381,
                evaluate_points_batch_bls12_381,
                "EC NTT",
                false,
                20,
            );
            bench_template(
                MAX_POINTS_LOG2,
                ntt_size,
                batch_size,
                log_ntt_size,
                set_up_points_bls12_381,
                interpolate_points_batch_bls12_381,
                "EC iNTT",
                true,
                20,
            );
        }
    }
}

fn bench_template<E, S>(
    log_max_size: usize,
    ntt_size: usize,
    batch_size: usize,
    log_ntt_size: usize,
    set_data: fn(
        test_size: usize,
        log_domain_size: usize,
        inverse: bool,
    ) -> (Vec<E>, DeviceBuffer<E>, DeviceBuffer<S>),
    bench_fn: fn(
        d_evaluations: &mut DeviceBuffer<E>,
        d_domain: &mut DeviceBuffer<S>,
        batch_size: usize,
    ) -> DeviceBuffer<E>,
    id: &str,
    inverse: bool,
    _samples: usize,
) -> Option<(Vec<E>, DeviceBuffer<E>)> {
    let count = ntt_size * batch_size;

    let bench_id = format!("{} of size 2^{} in batch {}", id, log_ntt_size, batch_size);

    if count > 1 << log_max_size {
        println!("Bench size exceeded: {}", bench_id);
        return None;
    }

    println!("{}", bench_id);

    let (input, mut d_evals, mut d_domain) = set_data(ntt_size * batch_size, log_ntt_size, inverse);
    range_push!("one entry");
    let first = bench_fn(&mut d_evals, &mut d_domain, batch_size);
    range_pop!();
    Some((input, first))
}

fn main () {
    bench_lde();
}
