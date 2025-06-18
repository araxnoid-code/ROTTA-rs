use crate::rotta_rs::Arrayy;

pub fn ln_arr(x: &Arrayy) -> Arrayy {
    let vector = x.value
        .iter()
        .map(|v| v.ln())
        .collect::<Vec<f64>>();

    Arrayy::from_vector(x.shape.clone(), vector)
}
