use crate::rotta_rs_module::arrayy::*;

pub fn exp_arr(arr: &Arrayy) -> Arrayy {
    let vector = arr.value
        .iter()
        .map(|v| { v.exp() })
        .collect::<Vec<f32>>();

    Arrayy::from_vector(arr.shape.clone(), vector)
}
