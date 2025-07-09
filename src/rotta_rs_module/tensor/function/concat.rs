use crate::{ arrayy::{ concat_arr, ArrSlice, Arrayy }, BackwardLabel, NodeType, Tensor };

pub fn concat(tensors: Vec<&Tensor>, dim: usize) -> Tensor {
    let mut check_shape = None;
    let mut nodes = Vec::new();
    let arrayys = tensors
        .into_iter()
        .map(|tensor| {
            if let None = check_shape {
                check_shape = Some(tensor.node.lock().unwrap().value.shape.clone());
            } else {
                if check_shape.as_ref().unwrap() != &tensor.node.lock().unwrap().value.shape {
                    panic!("concat error: shape of tensors not same");
                }
            }

            nodes.push(tensor.node.clone());
            tensor.value()
        })
        .collect::<Vec<Arrayy>>();

    let tensor = Tensor::from_arrayy(concat_arr(arrayys, dim));
    tensor.update_parent_label(nodes.clone(), Some(BackwardLabel::Concat(nodes, dim)));
    tensor
}

pub fn d_concat(nodes: Vec<NodeType>, dim: usize, grad: &Arrayy) {
    for (i, node) in nodes.iter().enumerate() {
        let mut node = node.lock().unwrap();
        let shape = &node.value.shape;
        let dim_len = shape[dim];

        let slice = shape[..dim + 1]
            .into_iter()
            .enumerate()
            .map(|(ii, _)| {
                let start = i * dim_len;
                let stop = start + dim_len;

                if ii == dim {
                    ArrSlice(Some(start as i32), Some(stop as i32))
                } else {
                    ArrSlice(None, None)
                }
            })
            .collect::<Vec<ArrSlice>>();

        node.add_grad(grad.slice(slice));
    }
}
