use crate::rotta_rs::{
    broadcast_concat,
    broadcasting_tensor_non_panic,
    Arrayy,
    BackwardLabel,
    NodeType,
    Tensor,
};

// add
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let broadcast_shape = broadcast_concat(&a.value(), &b.value());

    let broadcast_a = broadcasting_tensor_non_panic(a, broadcast_shape.clone());
    let broadcast_b = broadcasting_tensor_non_panic(b, broadcast_shape);

    // let broadcast_a = a;
    // let broadcast_b = b;

    let output = broadcast_a.value() + broadcast_b.value();
    let tensor = Tensor::from_arrayy(output);
    tensor.update_parent(vec![broadcast_a.node.clone(), broadcast_b.node.clone()]);
    tensor.node.lock().as_mut().unwrap().label = Some(
        BackwardLabel::Add(broadcast_a.node.clone(), broadcast_b.node.clone())
    );

    tensor
}

pub fn d_add(a: &NodeType, b: &NodeType, grad: Arrayy) {
    // f/da = 1 * grad = grad
    a.lock().as_mut().unwrap().add_grad(grad.clone());

    // f/db = 1 * grad = grad
    b.lock().as_mut().unwrap().add_grad(grad);
}
