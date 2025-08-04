use crate::Tensor;

pub fn mean_axis(x: &Tensor, d: &[i32]) -> Tensor {
    let shape = x.shape();

    let mut len = 1;
    d.iter().for_each(|d| {
        let d = (if *d >= 0 { *d } else { (shape.len() as i32) + d }) as usize;

        len *= shape[d];
    });

    &x.sum_axis(d) / (len as f32)
}

pub fn mean_axis_keep_dim(x: &Tensor, d: &[i32]) -> Tensor {
    let shape = x.shape();

    let mut len = 1;
    d.iter().for_each(|d| {
        let d = (if *d >= 0 { *d } else { (shape.len() as i32) + d }) as usize;

        len *= shape[d];
    });

    &x.sum_axis_keep_dim(d) / (len as f32)
}
