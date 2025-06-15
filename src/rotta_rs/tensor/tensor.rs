use std::{ fmt::Display, sync::{ Arc, Mutex } };

use ndarray::array;

use crate::rotta_rs::{ ArrayType, Arrayy, NdArray, Node, NodeType };

#[derive(Debug)]
pub struct Tensor {
    pub node: NodeType,
}

impl Tensor {
    // initialization
    pub fn new(array: Arrayy) -> Tensor {
        let node = Node::new(array);
        let node = Arc::new(Mutex::new(node));

        let tensor = Tensor {
            node,
        };

        tensor
    }

    // get
    // value
    pub fn value_vector(&self) -> Arrayy {
        self.node.lock().unwrap().value.clone()
    }

    // grad
    pub fn value_grad(&self) -> Arrayy {
        self.node.lock().unwrap().grad.clone()
    }

    // update
    // parent
    pub fn update_parent(&self, parent: Vec<NodeType>) {
        self.node.lock().as_mut().unwrap().parent = parent;
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let string = format!("{}", self.node.lock().unwrap().value);
        f.write_str(&string)
    }
}
