use crate::rotta_rs_module::arrayy::*;

pub fn powi_arr(arr: &Arrayy, n: i32) -> Arrayy {
    let vector = arr.value
        .iter()
        .map(|v| v.powi(n))
        .collect::<Vec<f32>>();

    Arrayy::from_vector(arr.shape.clone(), vector)
}

pub fn powf_arr(arr: &Arrayy, n: f32) -> Arrayy {
    let vector = arr.value
        .iter()
        .map(|v| v.powf(n))
        .collect::<Vec<f32>>();

    Arrayy::from_vector(arr.shape.clone(), vector)
}
