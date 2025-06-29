use crate::rotta_rs_module::{ arrayy::Arrayy, BackwardLabel, NodeType, Tensor };

pub fn sum(x: &Tensor) -> Tensor {
    let float = x.value().sum();

    let tensor = Tensor::new([float]);
    tensor.update_parent(vec![x.node.clone()]);
    tensor.update_label(Some(BackwardLabel::Sum(x.node.clone())));

    tensor
}

pub fn d_sum(x: &NodeType, grad: &Arrayy) {
    let mut x = x.lock().unwrap();

    if x.requires_grad {
        let d_x = Arrayy::ones(x.value.shape.clone()) * grad;

        x.add_grad(d_x);
    }
}
