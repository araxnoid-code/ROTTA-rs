use crate::rotta_rs_module::arrayy::*;

pub fn reshape_arr(x: &Arrayy, reshape: Vec<i32>) -> Arrayy {
    let mut minus_index = None;
    let mut shape = reshape
        .into_iter()
        .enumerate()
        .map(|(i, d)| {
            if d > 0 {
                d as usize
            } else if d == -1 {
                if let None = minus_index {
                    minus_index = Some(i);
                    1
                } else {
                    panic!("reshape only admit one negative value")
                }
            } else {
                panic!("reshape out of shape")
            }
        })
        .collect::<Vec<usize>>();

    if let Some(idx) = minus_index {
        shape[idx] = x.len() / shape.multiple_sum();
    }

    x.to_shape(shape)
}
