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
    // initialization
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

    // gradient
    pub fn ones_grad(&mut self) {
        self.grad = ndarray::Array2::ones(self.grad.dim());
    }

    pub fn add_grad(&mut self, grad: &NdArray) {
        self.grad = &self.grad + grad;
    }

    pub fn zero_grad(&mut self) {
        self.grad = &self.grad * 0.0;
    }

    // weight
    pub fn update_value(&mut self, value: NdArray) {
        self.value = value;
    }
}
