use crate::{ arrayy::{ concat_arr, ArrSlice, Arrayy }, BackwardLabel, NodeType, Tensor };

pub fn concat(tensors: Vec<&Tensor>, dim: i32) -> Tensor {
    let mut check_shape = None;
    let mut nodes = Vec::new();

    let dim = if dim >= 0 {
        dim as usize
    } else {
        ((tensors[0].shape().len() as i32) + dim) as usize
    };

    let arrayys = tensors
        .into_iter()
        .map(|tensor| {
            if let None = check_shape {
                check_shape = Some(tensor.node.read().unwrap().value.shape.clone());
            } else {
                if check_shape.as_ref().unwrap() != &tensor.node.read().unwrap().value.shape {
                    panic!("concat error: shape of tensors not same");
                }
            }

            nodes.push(tensor.node.clone());
            tensor.value()
        })
        .collect::<Vec<Arrayy>>();

    let tensor = Tensor::from_arrayy(concat_arr(arrayys, dim as i32));
    tensor.update_parent_label(nodes.clone(), Some(BackwardLabel::Concat(nodes, dim)));
    tensor
}

pub fn d_concat(nodes: Vec<NodeType>, dim: usize, grad: &Arrayy) {
    // for i in &nodes {
    //     println!("{}", i.lock().unwrap().value);
    // }

    for (i, node) in nodes.iter().enumerate() {
        let _node = node.read().unwrap();

        let shape = &_node.value.shape;
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

        node.write().unwrap().add_grad(grad.slice(&slice));
    }
}

// trait
pub trait ConcatTensors {
    fn concat_tensor(&self, dim: i32) -> Tensor;
}

impl ConcatTensors for Vec<Tensor> {
    fn concat_tensor(&self, dim: i32) -> Tensor {
        let tensors = self;
        let mut check_shape = None;
        let mut nodes = Vec::new();

        let dim = if dim >= 0 {
            dim as usize
        } else {
            ((tensors[0].shape().len() as i32) + dim) as usize
        };

        let arrayys = tensors
            .into_iter()
            .map(|tensor| {
                if let None = check_shape {
                    check_shape = Some(tensor.node.read().unwrap().value.shape.clone());
                } else {
                    if check_shape.as_ref().unwrap() != &tensor.node.read().unwrap().value.shape {
                        panic!("concat error: shape of tensors not same");
                    }
                }

                nodes.push(tensor.node.clone());
                tensor.value()
            })
            .collect::<Vec<Arrayy>>();

        let tensor = Tensor::from_arrayy(concat_arr(arrayys, dim as i32));
        tensor.update_parent_label(nodes.clone(), Some(BackwardLabel::Concat(nodes, dim)));
        tensor
    }
}

impl ConcatTensors for Vec<&Tensor> {
    fn concat_tensor(&self, dim: i32) -> Tensor {
        let tensors = self;
        let mut check_shape = None;
        let mut nodes = Vec::new();

        let dim = if dim >= 0 {
            dim as usize
        } else {
            ((tensors[0].shape().len() as i32) + dim) as usize
        };

        let arrayys = tensors
            .into_iter()
            .map(|tensor| {
                if let None = check_shape {
                    check_shape = Some(tensor.node.read().unwrap().value.shape.clone());
                } else {
                    if check_shape.as_ref().unwrap() != &tensor.node.read().unwrap().value.shape {
                        panic!("concat error: shape of tensors not same");
                    }
                }

                nodes.push(tensor.node.clone());
                tensor.value()
            })
            .collect::<Vec<Arrayy>>();

        let tensor = Tensor::from_arrayy(concat_arr(arrayys, dim as i32));
        tensor.update_parent_label(nodes.clone(), Some(BackwardLabel::Concat(nodes, dim)));
        tensor
    }
}
