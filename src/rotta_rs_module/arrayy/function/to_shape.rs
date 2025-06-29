use crate::rotta_rs_module::arrayy::*;

pub fn to_shape_arr(arr: &Arrayy, reshape: Vec<usize>) -> Arrayy {
    let arr_length = arr.value.len();
    let reshape_length = reshape.as_slice().multiple_sum();
    if arr_length != reshape_length {
        panic!("error, array have length {} but reshape to {:?}", arr_length, reshape);
    }

    Arrayy::from_vector(reshape, arr.value.clone())
}
