use core::slice;
use std::time::SystemTime;

use rotta_rs::{ arrayy::{ slice_arr, slice_replace_arr, Arrayy }, * };

fn main() {
    let mut array = Arrayy::ones(vec![256, 256, 512]);
    // println!("{}", array);

    let replace = Arrayy::ones(vec![256, 256, 512]);
    let slicing = vec![r(..), r(..), r(..)];

    let tick = std::time::SystemTime
        ::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis();

    slice_replace_arr(&mut array, &slicing, &replace);

    let tock = std::time::SystemTime
        ::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis();

    println!("{}ms", tock - tick)
}
