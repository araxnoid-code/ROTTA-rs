use crate::Tensor;

pub fn mean_axis(x: &Tensor, d: i32) -> Tensor {
    let shape = x.shape();

    let d = if d >= 0 { d as usize } else { ((shape.len() as i32) + d) as usize };

    &x.sum_axis(d as i32) / (shape[d] as f64)
}

pub fn mean_axis_keep_dim(x: &Tensor, d: i32) -> Tensor {
    let shape = x.shape();

    let d = if d >= 0 { d as usize } else { ((shape.len() as i32) + d) as usize };

    &x.sum_axis_keep_dim(d as i32) / (shape[d] as f64)
}
