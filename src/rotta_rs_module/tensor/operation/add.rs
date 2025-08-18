use std::ops::Add;

use crate::{
    rotta_rs_module::{
        arrayy::{ broadcast_concat, Arrayy, MultipleSum },
        broadcasting_tensor_non_panic,
        BackwardLabel,
        Tensor,
    },
    ShareTensor,
};

// add
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let a_arr = a.value.read().unwrap();
    let b_arr = b.value.read().unwrap();

    if a_arr.shape.multiple_sum() == 1 || b_arr.shape.multiple_sum() == 1 {
        // skalar
        let output = &*a_arr + &*b_arr;

        let mut tensor = Tensor::from_arrayy(output);
        tensor.update_parent(vec![a.shared_tensor(), b.shared_tensor()]);
        tensor.update_label(Some(BackwardLabel::Add(a.shared_tensor(), b.shared_tensor())));

        tensor
    } else if a_arr.shape == b_arr.shape {
        // same shape
        let output = &*a_arr + &*b_arr;

        let mut tensor = Tensor::from_arrayy(output);
        tensor.update_parent(vec![a.shared_tensor(), b.shared_tensor()]);
        tensor.update_label(Some(BackwardLabel::Add(a.shared_tensor(), b.shared_tensor())));

        tensor
    } else {
        // broadcasting
        let broadcast_shape = broadcast_concat(&a_arr, &b_arr);

        let broadcast_a = broadcasting_tensor_non_panic(a, broadcast_shape.clone());
        let broadcast_b = broadcasting_tensor_non_panic(b, broadcast_shape);

        let output = &*broadcast_a.value.read().unwrap() + &*broadcast_b.value.read().unwrap();
        let mut tensor = Tensor::from_arrayy(output);
        tensor.update_parent(vec![broadcast_a.shared_tensor(), broadcast_b.shared_tensor()]);
        tensor.update_label(
            Some(BackwardLabel::Add(broadcast_a.shared_tensor(), broadcast_b.shared_tensor()))
        );

        tensor
    }
}

pub fn d_add(a: &ShareTensor, b: &ShareTensor, grad: &Arrayy) {
    let tensor_a = a.value.read().unwrap();
    // let tensor_b = a.value.read().unwrap();

    // f/da = 1 * grad = grad
    if a.requires_grad() {
        let d_a = if tensor_a.shape.multiple_sum() == 1 {
            Arrayy::from_vector(tensor_a.shape.clone(), vec![grad.sum()])
        } else {
            grad.clone()
        };

        a.add_grad(d_a);
    }

    if b.requires_grad() {
        let d_b = if b.value.read().unwrap().shape.multiple_sum() == 1 {
            Arrayy::from_vector(b.value.read().unwrap().shape.clone(), vec![grad.sum()])
        } else {
            grad.clone()
        };

        b.add_grad(d_b);
    }
}

// method

impl Add<&Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Self::Output {
        add(self, rhs)
    }
}

impl Add<f32> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: f32) -> Self::Output {
        let rhs = Tensor::from_vector(vec![1], vec![rhs]);
        rhs.set_requires_grad(false);
        add(self, &rhs)
    }
}

impl Add<&Tensor> for f32 {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Self::Output {
        let float = Tensor::from_vector(vec![1], vec![self]);
        float.set_requires_grad(false);
        add(&float, rhs)
    }
}
