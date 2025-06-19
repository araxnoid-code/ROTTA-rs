use crate::rotta_rs::Arrayy;

pub fn powi_arr(arr: &Arrayy, n: i32) -> Arrayy {
    let vector = arr.value
        .iter()
        .map(|v| v.powi(n))
        .collect::<Vec<f64>>();

    Arrayy::from_vector(arr.shape.clone(), vector)
}

pub fn powf_arr(arr: &Arrayy, n: f64) -> Arrayy {
    let vector = arr.value
        .iter()
        .map(|v| v.powf(n))
        .collect::<Vec<f64>>();

    Arrayy::from_vector(arr.shape.clone(), vector)
}
