use crate::arrayy::Arrayy;

pub fn sin_arr(arr: &Arrayy) -> Arrayy {
    let vector = arr.value
        .iter()
        .map(|x| { x.sin() })
        .collect::<Vec<f32>>();

    Arrayy::from_vector(arr.shape.clone(), vector)
}

pub fn cos_arr(arr: &Arrayy) -> Arrayy {
    let vector = arr.value
        .iter()
        .map(|x| { x.cos() })
        .collect::<Vec<f32>>();

    Arrayy::from_vector(arr.shape.clone(), vector)
}

pub fn tan_arr(arr: &Arrayy) -> Arrayy {
    let vector = arr.value
        .iter()
        .map(|x| { x.tan() })
        .collect::<Vec<f32>>();

    Arrayy::from_vector(arr.shape.clone(), vector)
}
