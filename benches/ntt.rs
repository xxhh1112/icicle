extern crate criterion;

use criterion::{criterion_group, criterion_main, Criterion};

use icicle_utils::test_bls12_381::{
    interpolate_points_batch_bls12_381, interpolate_scalars_batch_bls12_381,
    set_up_points_bls12_381, set_up_scalars_bls12_381, evaluate_points_batch_bls12_381, evaluate_scalars_batch_bls12_381,
};
use rustacuda::prelude::DeviceBuffer;

const LOG_NTT_SIZES: [usize; 8] = [24, 26, 8, 9, 10, 11, 12, 18];
const BATCH_SIZES: [usize; 8] = [1, 4, 8, 16, 256, 512, 1024, 1 << 14];

const MAX_POINTS_LOG2: usize = 18;
const MAX_SCALARS_LOG2: usize = 24;

fn bench_ntt(c: &mut Criterion) {
    for log_ntt_size in LOG_NTT_SIZES {
        for batch_size in BATCH_SIZES {
            let ntt_size = 1 << log_ntt_size;

            bench_template(
                MAX_POINTS_LOG2,
                ntt_size,
                batch_size,
                log_ntt_size,
                set_up_points_bls12_381,
                evaluate_points_batch_bls12_381,
                c,
                "EC NTT",
                false
            );
            bench_template(
                MAX_POINTS_LOG2,
                ntt_size,
                batch_size,
                log_ntt_size,
                set_up_points_bls12_381,
                interpolate_points_batch_bls12_381,
                c,
                "EC iNTT",
                true
            );
            bench_template(
                MAX_SCALARS_LOG2,
                ntt_size,
                batch_size,
                log_ntt_size,
                set_up_scalars_bls12_381,
                evaluate_scalars_batch_bls12_381,
                c,
                "NTT",
                false
            );
            bench_template(
                MAX_SCALARS_LOG2,
                ntt_size,
                batch_size,
                log_ntt_size,
                set_up_scalars_bls12_381,
                interpolate_scalars_batch_bls12_381,
                c,
                "iNTT",
                true
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
    c: &mut Criterion,
    id: &str,
    inverse: bool,
) {
    let count = ntt_size * batch_size;

    let bench_id = &format!("{} of size 2^{} in batch {}", id, log_ntt_size, batch_size);

    if count > 1 << log_max_size {
        println!("Bench size exceeded: {}", bench_id);
        return;
    }

    let (_, mut d_evals, mut d_domain) = set_data(ntt_size * batch_size, log_ntt_size, inverse);

    c.bench_function(&bench_id, |b| {
        b.iter(|| bench_fn(&mut d_evals, &mut d_domain, batch_size))
    });
}

criterion_group!(ntt_benches, bench_ntt);
criterion_main!(ntt_benches);
