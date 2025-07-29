use crate::rotta_rs_module::{ arrayy::Arrayy, BackwardLabel, NodeType, Tensor };

pub fn powf(x: &Tensor, n: f64) -> Tensor {
    let arrayy = x.value().powf(n);
    let tensor = Tensor::from_arrayy(arrayy);
    tensor.update_parent(vec![x.node.clone()]);
    tensor.update_label(Some(BackwardLabel::Powf(x.node.clone(), n)));

    tensor
}

pub fn d_powf(x: &NodeType, powf: f64, grad: &Arrayy) {
    let mut _x = x.write().unwrap();

    // d/x = n * x^n-1
    if _x.requires_grad {
        let dx = powf * &_x.value.powf(powf - 1.0) * grad;

        _x.add_grad(dx);
    }
}
