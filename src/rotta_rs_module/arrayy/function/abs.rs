use crate::rotta_rs_module::arrayy::*;

#[allow(dead_code)]
pub fn abs_arr(x: &Arrayy) -> Arrayy {
    let vector = x.value
        .iter()
        .map(|v| v.abs())
        .collect::<Vec<f32>>();

    Arrayy::from_vector(x.shape.clone(), vector)
}
