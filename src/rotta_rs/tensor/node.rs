use uuid::Uuid;

use crate::rotta_rs::{ Arrayy, BackwardLabel, NodeType };

#[derive(Debug)]
pub struct Node {
    pub id: u128,
    pub value: Arrayy,
    pub grad: Arrayy,
    pub parent: Vec<NodeType>,
    pub label: Option<BackwardLabel>,
    // requires
    pub requires_grad: bool,
    pub auto_zero_grad: bool,
}

impl Node {
    // initialization
    pub fn new(value: Arrayy) -> Node {
        let node = Node {
            id: Uuid::new_v4().as_u128(),
            grad: Arrayy::zeros(value.shape.clone()),
            value,
            parent: Vec::new(),
            label: None,
            requires_grad: true,
            auto_zero_grad: true,
        };

        node
    }

    // gradient
    pub fn ones_grad(&mut self) {
        self.grad = Arrayy::ones(self.value.shape.clone());
    }

    pub fn add_grad(&mut self, grad: Arrayy) {
        if self.requires_grad {
            self.grad = self.grad.clone() + grad;
        }
    }

    pub fn zero_grad(&mut self) {
        self.grad = self.grad.clone() * Arrayy::zeros(vec![1]);
    }

    // weight
    pub fn update_value(&mut self, value: Arrayy) {
        self.value = value;
    }
}
