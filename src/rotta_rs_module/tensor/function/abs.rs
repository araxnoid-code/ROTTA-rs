use crate::{ rotta_rs_module::{ arrayy::Arrayy, BackwardLabel, Tensor }, ShareTensor };

pub fn abs(x: &Tensor) -> Tensor {
    let mut tensor = Tensor::from_arrayy(x.value.read().unwrap().abs());
    tensor.update_parent(vec![x.shared_tensor()]);
    tensor.update_label(Some(BackwardLabel::Abs(x.shared_tensor())));

    tensor
}

pub fn d_abs(x: &ShareTensor, grad: &Arrayy) {
    if x.requires_grad() {
        let d_x = x.value.read().unwrap().sign() * grad;
        x.add_grad(d_x);
    }
}
