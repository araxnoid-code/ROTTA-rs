use crate::arrayy::{ sum_axis_arr, Arrayy };

pub fn mean_axis_arr(arr: &Arrayy, d: i32) -> Arrayy {
    let shape = &arr.shape;
    let d = if d >= 0 { d as usize } else { ((shape.len() as i32) + d) as usize };

    arr.sum_axis(d as i32) / (shape[d] as f64)
}

pub fn mean_axis_keep_dim_arr(arr: &Arrayy, d: i32) -> Arrayy {
    let shape = &arr.shape;
    let d = if d >= 0 { d as usize } else { ((shape.len() as i32) + d) as usize };

    arr.sum_axis_keep_dim(d as i32) / (shape[d] as f64)
}
