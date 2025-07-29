use crate::rotta_rs_module::{ arrayy::Arrayy, BackwardLabel, NodeType, Tensor };

pub fn sum(x: &Tensor) -> Tensor {
    let float = x.value().sum();

    let tensor = Tensor::new([float]);
    tensor.update_parent(vec![x.node.clone()]);
    tensor.update_label(Some(BackwardLabel::Sum(x.node.clone())));

    tensor
}

pub fn d_sum(x: &NodeType, grad: &Arrayy) {
    let mut _x = x.write().unwrap();

    if _x.requires_grad {
        let d_x = Arrayy::ones(_x.value.shape.clone()) * grad;

        _x.add_grad(d_x);
    }
}
