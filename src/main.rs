use std::time::{ SystemTime, UNIX_EPOCH };

use crate::rotta_rs::{ sum_arr, Arrayy, Tensor };

mod rotta_rs;

fn main() {
    let tensor = Tensor::new([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]);

    let skalar = Tensor::new([[[1.0]]]);

    (&tensor + &skalar).backward();

    println!("{}", skalar.grad())
}
