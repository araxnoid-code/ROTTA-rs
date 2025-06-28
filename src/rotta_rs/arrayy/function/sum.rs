use crate::rotta_rs::arrayy::*;

pub fn sum_arr(arr: &Arrayy) -> f64 {
    arr.value.iter().sum::<f64>()
}
