use crate::{ rotta_rs_module::{ arrayy::Arrayy, BackwardLabel, NodeType, Tensor }, ShareTensor };

pub fn powf(x: &Tensor, n: f64) -> Tensor {
    let arrayy = x.value.read().unwrap().powf(n);
    let mut tensor = Tensor::from_arrayy(arrayy);
    tensor.update_parent(vec![x.shared_tensor()]);
    tensor.update_label(Some(BackwardLabel::Powf(x.shared_tensor(), n)));

    tensor
}

pub fn d_powf(x: &ShareTensor, powf: f64, grad: &Arrayy) {
    // d/x = n * x^n-1
    if x.requires_grad() {
        let dx =
            powf *
            &x.value
                .read()
                .unwrap()
                .powf(powf - 1.0) *
            grad;

        x.add_grad(dx);
    }
}
