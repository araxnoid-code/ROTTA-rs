use crate::{
    arrayy::sum_axis_keep_dim_arr,
    rotta_rs_module::{ arrayy::{ sum_axis_arr, to_shape_arr, Arrayy }, BackwardLabel, Tensor },
    ShareTensor,
};

pub fn sum_axis(x: &Tensor, d: &[i32]) -> Tensor {
    let array = x.value.read().unwrap();
    let mut tensor = Tensor::from_arrayy(sum_axis_arr(&array, d));
    tensor.update_parent(vec![x.shared_tensor()]);
    tensor.update_label(
        Some(
            BackwardLabel::SumAxis(
                x.shared_tensor(),
                {
                    let mut vec = d.to_vec();
                    vec.sort();
                    vec
                },
                false
            )
        )
    );

    tensor
}

pub fn sum_axis_keep_dim(x: &Tensor, d: &[i32]) -> Tensor {
    let array = x.value.read().unwrap();

    let sum = sum_axis_keep_dim_arr(&array, d);

    let mut tensor = Tensor::from_arrayy(sum);
    tensor.update_parent(vec![x.shared_tensor()]);
    tensor.update_label(Some(BackwardLabel::SumAxis(x.shared_tensor(), d.to_vec(), true)));

    tensor
}

pub fn d_sum_axis(x: &ShareTensor, d: &[i32], keep_dim: bool, grad: &Arrayy) {
    if x.requires_grad() {
        if !keep_dim {
            let ones = Arrayy::ones(x.value.read().unwrap().shape.clone());
            let mut new_shape = grad.shape.clone();

            for d in d {
                new_shape.insert(*d as usize, 1);
            }

            let d = ones * to_shape_arr(grad, new_shape);
            x.add_grad(d);
        } else {
            let ones = Arrayy::ones(x.value.read().unwrap().shape.clone());
            // println!("{}", grad);
            let d = ones * grad;
            x.add_grad(d);
        }
    }
}
