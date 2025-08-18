use crate::{ rotta_rs_module::{ arrayy::Arrayy, BackwardLabel, Tensor }, ShareTensor };

pub fn sum(x: &Tensor) -> Tensor {
    let float = x.value.read().unwrap().sum();

    let mut tensor = Tensor::new([float]);
    tensor.update_parent(vec![x.shared_tensor()]);
    tensor.update_label(Some(BackwardLabel::Sum(x.shared_tensor())));

    tensor
}

pub fn d_sum(x: &ShareTensor, grad: &Arrayy) {
    if x.requires_grad() {
        let d_x = Arrayy::ones(x.value.read().unwrap().shape.clone()) * grad;

        x.add_grad(d_x);
    }
}
