use std::time::SystemTime;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::rngs::StdRng;
use rand_distr::{ Distribution, Normal };

fn main() {
    let mut chacha = ChaCha8Rng::seed_from_u64(42);
    let mut std_rng = StdRng::seed_from_u64(42);

    let normal = Normal::new(0.0, 1.0).unwrap();

    let tick = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos();
    for _ in 0..100000 {
        let _data = normal.sample(&mut chacha);
        // let _data = normal.sample(&mut std_rng);
    }
    let tock = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos();
    println!("{}ms", tock - tick);
}
