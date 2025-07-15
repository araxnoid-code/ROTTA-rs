use core::slice;
use std::time::SystemTime;

use rotta_rs::{ arrayy::{ slice_arr, slice_arr_unsafe, Arrayy }, * };

fn main() {
    let array = Arrayy::arange(0, 24, 1).reshape(vec![4, 3, 2]);
    println!("{}", array);

    let slicing = slice_arr_unsafe(&array, &[r(2..), r(2..3), r(..1)]);
    println!("{}", slicing)

    // let slicing = vec![r(..), r(..), r(..)];
    // let tick = std::time::SystemTime
    //     ::now()
    //     .duration_since(SystemTime::UNIX_EPOCH)
    //     .unwrap()
    //     .as_millis();

    // let unsafe_slice = slice_arr_unsafe(&array, &slicing);
    // // let slice = slice_arr(&array, slicing);

    // let tock = std::time::SystemTime
    //     ::now()
    //     .duration_since(SystemTime::UNIX_EPOCH)
    //     .unwrap()
    //     .as_millis();

    // println!("{}ms", tock - tick)
}
