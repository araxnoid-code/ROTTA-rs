use crate::{ rotta_rs_module::{ arrayy::Arrayy, BackwardLabel, NodeType, Tensor }, ShareTensor };

pub fn powi(x: &Tensor, n: i32) -> Tensor {
    let arrayy = x.value.read().unwrap().powi(n);
    let mut tensor = Tensor::from_arrayy(arrayy);
    tensor.update_parent(vec![x.shared_tensor()]);
    tensor.update_label(Some(BackwardLabel::Powi(x.shared_tensor(), n)));

    tensor
}

pub fn d_powi(x: &ShareTensor, powi: i32, grad: &Arrayy) {
    // d/x = n * x^n-1
    if x.requires_grad() {
        let dx =
            (powi as f32) *
            &x.value
                .read()
                .unwrap()
                .powi(powi - 1) *
            grad;

        x.add_grad(dx);
    }
}
