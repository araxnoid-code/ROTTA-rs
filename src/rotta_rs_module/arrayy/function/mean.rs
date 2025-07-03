use crate::arrayy::Arrayy;

pub fn mean_arr(arr: &Arrayy) -> Arrayy {
    let mean = arr.value.iter().sum::<f64>() / (arr.value.len() as f64);
    Arrayy::new([mean])
}
