use std::time::{ SystemTime, UNIX_EPOCH };

use crate::rotta_rs::{ sum_arr, Arrayy, Tensor };

mod rotta_rs;

fn main() {
    let tensor = Tensor::new([
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0],
    ]);

    let skalar = Tensor::new([25.0]);

    (&skalar / &tensor).backward();
    println!("{}", skalar.grad());
}
