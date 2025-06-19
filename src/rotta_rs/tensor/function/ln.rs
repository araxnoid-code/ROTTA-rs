use crate::rotta_rs::{ Arrayy, BackwardLabel, NodeType, Tensor };

pub fn ln(x: &Tensor) -> Tensor {
    let tensor = Tensor::from_arrayy(x.value().ln());
    tensor.update_parent(vec![x.node.clone()]);
    tensor.update_label(Some(BackwardLabel::Ln(x.node.clone())));

    tensor
}

pub fn d_ln(x: &NodeType, grad: &Arrayy) {
    // dx = 1/x
    let dx = (1.0 / &x.lock().unwrap().value) * grad;
    x.lock().unwrap().add_grad(dx);
}
