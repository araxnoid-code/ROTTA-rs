use crate::{ rotta_rs_module::{ BackwardLabel, NodeType, Tensor }, ShareTensor };

pub fn sign(x: &Tensor) -> Tensor {
    let mut tensor = Tensor::from_arrayy(x.value.read().unwrap().sign());
    tensor.update_parent(vec![x.shared_tensor()]);
    tensor.update_label(Some(BackwardLabel::Sign(x.shared_tensor())));

    tensor
}

pub fn d_sign(x: &ShareTensor) {
    // dx = 0 * grad = 0
    x.zero_grad();
}
