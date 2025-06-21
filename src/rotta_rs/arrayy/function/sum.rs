use crate::rotta_rs::Arrayy;

pub fn sum_arr(arr: &Arrayy) -> f64 {
    arr.value.iter().sum::<f64>()
}
