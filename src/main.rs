use std::time::SystemTime;

use crate::rotta_rs::{ sum_arr, Arrayy };

mod rotta_rs;

fn main() {
    let arr = Arrayy::from_vector(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    println!("{}", arr.sum())
}
