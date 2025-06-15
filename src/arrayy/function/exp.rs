use crate::Arrayy;

pub fn exp(arr: &Arrayy) -> Arrayy {
    let vector = arr.value
        .iter()
        .map(|v| { v.exp() })
        .collect::<Vec<f64>>();

    Arrayy::from_vector(arr.shape.clone(), vector)
}
