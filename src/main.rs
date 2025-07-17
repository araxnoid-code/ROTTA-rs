use std::time::SystemTime;

use rotta_rs::Tensor;

fn main() {
    let length = 1_000_000_00;
    let tensor_a = Tensor::from_vector(vec![length], vec![0.0;length]);
    let tensor_b = Tensor::from_vector(vec![length], vec![100.0;length]);

    let tick = std::time::SystemTime
        ::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis();

    let _ = &tensor_a + &tensor_b;

    let tock = std::time::SystemTime
        ::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis();

    println!("{}ms", tock - tick);
}
