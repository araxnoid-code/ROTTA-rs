use crate::rotta_rs::arrayy::*;

// function
pub fn minus(arr_a: &Arrayy, arr_b: &Arrayy) -> Arrayy {
    let (vector, shape) = sub_arr_slice((&arr_a.value, &arr_a.shape), (&arr_b.value, &arr_b.shape));
    Arrayy::from_vector(shape, vector)
}
