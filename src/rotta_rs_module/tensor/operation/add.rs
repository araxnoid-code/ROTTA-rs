use std::ops::Add;

use crate::rotta_rs_module::{
    arrayy::broadcast_concat,
    broadcasting_tensor_non_panic,
    arrayy::Arrayy,
    BackwardLabel,
    arrayy::MultipleSum,
    NodeType,
    Tensor,
};

// add
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let a_arr = a.value();
    let b_arr = b.value();

    if a_arr.shape.multiple_sum() == 1 || b_arr.shape.multiple_sum() == 1 {
        // skalar
        let output = a_arr + b_arr;

        let tensor = Tensor::from_arrayy(output);
        tensor.update_parent(vec![a.node.clone(), b.node.clone()]);
        tensor.node.write().unwrap().label = Some(
            BackwardLabel::Add(a.node.clone(), b.node.clone())
        );

        tensor
    } else if a_arr.shape == b_arr.shape {
        // same shape
        let output = a_arr + b_arr;

        let tensor = Tensor::from_arrayy(output);
        tensor.update_parent(vec![a.node.clone(), b.node.clone()]);
        tensor.node.write().unwrap().label = Some(
            BackwardLabel::Add(a.node.clone(), b.node.clone())
        );

        tensor
    } else {
        // broadcasting
        let broadcast_shape = broadcast_concat(&a.value(), &b.value());

        let broadcast_a = broadcasting_tensor_non_panic(a, broadcast_shape.clone());
        let broadcast_b = broadcasting_tensor_non_panic(b, broadcast_shape);

        let output = broadcast_a.value() + broadcast_b.value();
        let tensor = Tensor::from_arrayy(output);
        tensor.update_parent(vec![broadcast_a.node.clone(), broadcast_b.node.clone()]);
        tensor.node.write().unwrap().label = Some(
            BackwardLabel::Add(broadcast_a.node.clone(), broadcast_b.node.clone())
        );

        tensor
    }
}

pub fn d_add(a: &NodeType, b: &NodeType, grad: Arrayy) {
    let _a = a.read().unwrap();
    let _b = b.read().unwrap();

    // f/da = 1 * grad = grad
    if _a.requires_grad {
        let d_a = if _a.value.shape.multiple_sum() == 1 {
            Arrayy::from_vector(_a.value.shape.clone(), vec![grad.sum()])
        } else {
            grad.clone()
        };
        a.write().unwrap().add_grad(d_a);
    }

    // f/db = 1 * grad = grad
    if _b.requires_grad {
        let d_b = if _b.value.shape.multiple_sum() == 1 {
            Arrayy::from_vector(_b.value.shape.clone(), vec![grad.sum()])
        } else {
            grad.clone()
        };
        b.write().unwrap().add_grad(d_b);
    }
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
        rhs.set_requires_grad(false);
        add(self, &rhs)
    }
}

impl Add<&Tensor> for f64 {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Self::Output {
        let float = Tensor::from_vector(vec![1], vec![self]);
        float.set_requires_grad(false);
        add(&float, rhs)
    }
}
