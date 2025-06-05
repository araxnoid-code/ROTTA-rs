use std::{ fmt::Display, sync::{ Arc, Mutex } };

use uuid::Uuid;

use crate::rotta_rs::{ BackwardLabel, NdArray, Node, NodeType };

#[derive(Debug)]
pub struct Tensor {
    pub node: NodeType,
}

impl Tensor {
    // initialization
    pub fn new(value: NdArray) -> Tensor {
        let node = Node::new(value);
        let node = Arc::new(Mutex::new(node));

        let tensor = Tensor {
            node,
        };

        tensor
    }

    // get
    // value
    pub fn value(&self) -> NdArray {
        self.node.lock().unwrap().value.clone()
    }

    // grad
    pub fn grad(&self) -> NdArray {
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
