use crate::arrayy::Arrayy;

pub fn mean_axis_arr(arr: &Arrayy, d: &[i32]) -> Arrayy {
    let shape = &arr.shape;

    let mut len = 1;
    d.iter().for_each(|d| {
        let d = (if *d >= 0 { *d } else { (shape.len() as i32) + d }) as usize;

        len *= shape[d];
    });

    arr.sum_axis(d) / (len as f64)
}

pub fn mean_axis_keep_dim_arr(arr: &Arrayy, d: &[i32]) -> Arrayy {
    let shape = &arr.shape;
    let mut len = 1;
    d.iter().for_each(|d| {
        let d = (if *d >= 0 { *d } else { (shape.len() as i32) + d }) as usize;

        len *= shape[d];
    });

    arr.sum_axis_keep_dim(d) / (len as f64)
}
