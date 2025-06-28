use crate::rotta_rs::arrayy::*;

pub fn exp_arr(arr: &Arrayy) -> Arrayy {
    let vector = arr.value
        .iter()
        .map(|v| { v.exp() })
        .collect::<Vec<f64>>();

    Arrayy::from_vector(arr.shape.clone(), vector)
}
