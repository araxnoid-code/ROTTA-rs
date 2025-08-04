use std::time::SystemTime;

use rotta_rs::{ matmul, Tensor };

fn main() {
    let a = Tensor::rand(vec![10000, 10000]);
    let b = Tensor::rand(vec![10000, 10000]);

    let tick = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis();

    matmul(&a, &b);

    let tock = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis();

    println!("{}ms", tock - tick);
}
