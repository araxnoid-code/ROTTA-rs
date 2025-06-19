use crate::rotta_rs::{
    broadcast_concat,
    broadcasting_tensor_non_panic,
    Arrayy,
    BackwardLabel,
    NodeType,
    Tensor,
};

pub fn divided(a: &Tensor, b: &Tensor) -> Tensor {
    let broadcast_shape = broadcast_concat(&a.value(), &b.value());

    let broadcast_a = broadcasting_tensor_non_panic(a, broadcast_shape.clone());
    let broadcast_b = broadcasting_tensor_non_panic(b, broadcast_shape);

    let tensor = Tensor::from_arrayy(broadcast_a.value() / broadcast_b.value());
    tensor.update_parent(vec![a.node.clone(), b.node.clone()]);
    tensor.node.lock().unwrap().label = Some(
        BackwardLabel::Diveded(a.node.clone(), b.node.clone())
    );

    tensor
}

pub fn d_divided(a: &NodeType, b: &NodeType, grad: &Arrayy) {
    let a_value = a.lock().unwrap().value.clone();
    let b_value = b.lock().unwrap().value.clone();

    // da = 1/b
    let da = (Arrayy::from_vector(vec![1], vec![1.0]) / &b_value) * grad;
    a.lock().unwrap().add_grad(da);

    // db = -a/b^2
    let db = -1.0 * (a_value / b_value.powi(2)) * grad;
    b.lock().unwrap().add_grad(db);
}
