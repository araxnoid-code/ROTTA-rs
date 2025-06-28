use crate::rotta_rs::{ arrayy::Arrayy, BackwardLabel, NodeType, Tensor };

pub fn sign(x: &Tensor) -> Tensor {
    let tensor = Tensor::from_arrayy(x.value().sign());
    tensor.update_parent(vec![x.node.clone()]);
    tensor.update_label(Some(BackwardLabel::Sign(x.node.clone())));

    tensor
}

pub fn d_sign(x: &NodeType) {
    // dx = 0 * grad = 0
    x.lock().unwrap().zero_grad();
}
