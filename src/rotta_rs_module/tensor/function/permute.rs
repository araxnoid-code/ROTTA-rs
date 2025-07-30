use crate::{ rotta_rs_module::{ arrayy::Arrayy, BackwardLabel, NodeType, Tensor }, ShareTensor };

pub fn permute(x: &Tensor, order: Vec<usize>) -> Tensor {
    let mut tensor = Tensor::from_arrayy(x.value.read().unwrap().permute(&order));
    tensor.update_parent(vec![x.shared_tensor()]);
    tensor.update_label(Some(BackwardLabel::Permute(x.shared_tensor(), order)));

    tensor
}

pub fn d_permute(x: &ShareTensor, order: Vec<usize>, grad: &Arrayy) {
    // let x = x.read().unwrap();

    let mut new_order = order.clone();
    for (i, d) in order.iter().enumerate() {
        new_order[*d] = i;
    }

    let d_x = grad.permute(&new_order);
    x.add_grad(d_x);
}
