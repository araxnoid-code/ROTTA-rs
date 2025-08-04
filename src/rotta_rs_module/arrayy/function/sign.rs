use crate::rotta_rs_module::arrayy::*;

pub fn sign_arr(x: &Arrayy) -> Arrayy {
    let vector = x.value
        .iter()
        .map(|v| if *v > 0.0 { 1.0 } else if *v < 0.0 { -1.0 } else { 0.0 })
        .collect::<Vec<f32>>();

    Arrayy::from_vector(x.shape.clone(), vector)
}
