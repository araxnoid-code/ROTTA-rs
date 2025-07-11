use std::time::SystemTime;

use rotta_rs::*;
use wide::f64x4;

fn main() {
    let data_a = vec![0.0;100004];
    let data_b = vec![0.0;100004];

    let tick = std::time::SystemTime
        ::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_micros();

    // let mut output = vec![0.0;data_a.len()];
    // // let mut output = vec![];

    // for i in 0..data_a.len() {
    //     output[i] = (data_a[i] as f64).abs();
    //     // output.push(data_a[i] + data_b[i]);
    // }

    let mut output = vec![];
    for i in (0..data_a.len()).step_by(4) {
        // let data = f64x4::
        output.extend_from_slice(
            f64x4
                ::from(&data_a[i..i + 4])
                .abs()
                .as_array_ref()
        );
    }

    let tock = std::time::SystemTime
        ::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_micros();

    println!("{}micro", tock - tick);
}
