use crate::arrayy::Arrayy;

pub fn mean_arr(arr: &Arrayy) -> Arrayy {
    let mean = arr.value.iter().sum::<f32>() / (arr.value.len() as f32);
    Arrayy::new([mean])
}
