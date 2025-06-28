use crate::rotta_rs::arrayy::*;

pub fn abs_arr(x: &Arrayy) -> Arrayy {
    let vector = x.value
        .iter()
        .map(|v| v.abs())
        .collect::<Vec<f64>>();

    Arrayy::from_vector(x.shape.clone(), vector)
}
