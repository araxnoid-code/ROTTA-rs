use std::ops::Add;

use crate::rotta_rs::{
    broadcast_concat,
    broadcasting_tensor_non_panic,
    Arrayy,
    BackwardLabel,
    MultipleSum,
    NodeType,
    Tensor,
};

// add
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let a_arr = a.value();
    let b_arr = b.value();

    // skalar
    if a_arr.shape.multiple_sum() == 1 || b_arr.shape.multiple_sum() == 1 {
        let output = a_arr + b_arr;

        let tensor = Tensor::from_arrayy(output);
        tensor.update_parent(vec![a.node.clone(), b.node.clone()]);
        tensor.node.lock().as_mut().unwrap().label = Some(
            BackwardLabel::Add(a.node.clone(), b.node.clone())
        );

        tensor
    } else if a_arr.shape == b_arr.shape {
        let output = a_arr + b_arr;

        let tensor = Tensor::from_arrayy(output);
        tensor.update_parent(vec![a.node.clone(), b.node.clone()]);
        tensor.node.lock().as_mut().unwrap().label = Some(
            BackwardLabel::Add(a.node.clone(), b.node.clone())
        );

        tensor
    } else {
        let broadcast_shape = broadcast_concat(&a.value(), &b.value());

        let broadcast_a = broadcasting_tensor_non_panic(a, broadcast_shape.clone());
        let broadcast_b = broadcasting_tensor_non_panic(b, broadcast_shape);

        let output = broadcast_a.value() + broadcast_b.value();
        let tensor = Tensor::from_arrayy(output);
        tensor.update_parent(vec![broadcast_a.node.clone(), broadcast_b.node.clone()]);
        tensor.node.lock().as_mut().unwrap().label = Some(
            BackwardLabel::Add(broadcast_a.node.clone(), broadcast_b.node.clone())
        );

        tensor
    }
}

pub fn d_add(a: &NodeType, b: &NodeType, grad: Arrayy) {
    // f/da = 1 * grad = grad

    let d_a = if a.lock().unwrap().value.shape.multiple_sum() == 1 {
        Arrayy::from_vector(a.lock().unwrap().value.shape.clone(), vec![grad.sum()])
    } else {
        grad.clone()
    };
    a.lock().as_mut().unwrap().add_grad(d_a);

    // f/db = 1 * grad = grad
    let d_b = if b.lock().unwrap().value.shape.multiple_sum() == 1 {
        Arrayy::from_vector(b.lock().unwrap().value.shape.clone(), vec![grad.sum()])
    } else {
        grad.clone()
    };
    b.lock().as_mut().unwrap().add_grad(d_b);
}

// method

impl Add<&Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Self::Output {
        add(self, rhs)
    }
}

impl Add<f64> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: f64) -> Self::Output {
        let rhs = Tensor::from_vector(vec![1], vec![rhs]);
        add(self, &rhs)
    }
}

impl Add<&Tensor> for f64 {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Self::Output {
        let float = Tensor::from_vector(vec![1], vec![self]);
        add(&float, rhs)
    }
}
