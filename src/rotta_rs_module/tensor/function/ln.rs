use crate::{ rotta_rs_module::{ arrayy::Arrayy, BackwardLabel, NodeType, Tensor }, ShareTensor };

pub fn ln(x: &Tensor) -> Tensor {
    let mut tensor = Tensor::from_arrayy(x.value.read().unwrap().ln());
    tensor.update_parent(vec![x.shared_tensor()]);
    tensor.update_label(Some(BackwardLabel::Ln(x.shared_tensor())));

    tensor
}

pub fn d_ln(x: &ShareTensor, grad: &Arrayy) {
    // dx = 1/x
    if x.requires_grad() {
        let dx = (1.0 / &*x.value.read().unwrap()) * grad;
        x.add_grad(dx);
    }
}
