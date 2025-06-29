use crate::rotta_rs_module::arrayy::*;

// function
pub fn mul_arr(arr_a: &Arrayy, arr_b: &Arrayy) -> Arrayy {
    let (vector, shape) = mul_arr_slice((&arr_a.value, &arr_a.shape), (&arr_b.value, &arr_b.shape));
    Arrayy::from_vector(shape, vector)
}
