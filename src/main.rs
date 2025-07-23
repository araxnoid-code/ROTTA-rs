use std::time::SystemTime;

use rotta_rs::Tensor;

fn main() {
    let tensor = Tensor::rand(vec![256, 256, 1024]);

    println!("{}", "bachmark running");
    let tick = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis();
    tensor.flatten();
    let tock = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis();

    println!("{}ms", tock - tick);
}
