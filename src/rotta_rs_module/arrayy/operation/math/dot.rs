use crate::rotta_rs_module::arrayy::*;

// function
pub fn dot_arr(arr_a: &Arrayy, arr_b: &Arrayy) -> Arrayy {
    let arr_a_s = arr_a.shape.clone();
    let arr_b_s = arr_b.shape.clone();

    if arr_a_s == arr_b_s && arr_a_s.len() == 1 && arr_b_s.len() == 1 {
        //
        let mut sum = 0.0;

        for (i, v) in arr_a.value.iter().enumerate() {
            sum += v * arr_b.value[i];
        }

        let array = Arrayy::from_vector(vec![1], vec![sum]);

        return array;
    } else {
        panic!(
            "error can't dot product cause differend shape, {:?} will dot product with {:?}",
            arr_a_s,
            arr_b_s
        )
    }
}
