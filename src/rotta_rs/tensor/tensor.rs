use std::{ fmt::Display, sync::{ Arc, Mutex } };

use uuid::Uuid;

use crate::rotta_rs::{ BackwardLabel, NdArray, NodeType };

#[derive(Debug)]
pub struct Node {
    pub id: u128,
    pub value: NdArray,
    pub grad: NdArray,
    pub parent: Vec<NodeType>,
    pub label: Option<BackwardLabel>,
}

impl Node {
    pub fn new(value: NdArray) -> Node {
        let node = Node {
            id: Uuid::new_v4().as_u128(),
            grad: ndarray::Array2::zeros(value.dim()),
            value,
            parent: Vec::new(),
            label: None,
        };

        node
    }

    pub fn ones_grad(&mut self) {
        self.grad = &self.grad + 1.0;
    }

    pub fn add_grad(&mut self, grad: &NdArray) {
        self.grad = &self.grad + grad;
    }
}

#[derive(Debug)]
pub struct Tensor {
    pub node: NodeType,
}

impl Tensor {
    pub fn new(value: NdArray) -> Tensor {
        let node = Node::new(value);
        let node = Arc::new(Mutex::new(node));

        let tensor = Tensor {
            node,
        };

        tensor
    }

    pub fn value(&self) -> NdArray {
        self.node.lock().unwrap().value.clone()
    }

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
