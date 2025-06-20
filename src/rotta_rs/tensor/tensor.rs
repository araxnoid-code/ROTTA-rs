use std::{ fmt::Display, ops::Mul, sync::{ Arc, Mutex } };

use crate::rotta_rs::{ Arrayy, BackwardLabel, Node, NodeType, RecFlatten };

#[derive(Debug)]
pub struct Tensor {
    pub node: NodeType,
}

impl Tensor {
    // initialization
    pub fn new<T: RecFlatten>(arr: T) -> Tensor {
        let (shape, vector) = arr.rec_flatten();
        let arr = Arrayy::from_vector(shape, vector);
        let node = Arc::new(Mutex::new(Node::new(arr)));

        Tensor {
            node,
        }
    }

    pub fn from_arrayy(array: Arrayy) -> Tensor {
        let node = Node::new(array);
        let node = Arc::new(Mutex::new(node));

        let tensor = Tensor {
            node,
        };

        tensor
    }

    pub fn from_vector(shape: Vec<usize>, vector: Vec<f64>) -> Tensor {
        let array = Arrayy::from_vector(shape, vector);

        let node = Node::new(array);
        let node = Arc::new(Mutex::new(node));

        let tensor = Tensor {
            node,
        };

        tensor
    }

    // get
    // value
    pub fn value(&self) -> Arrayy {
        self.node.lock().unwrap().value.clone()
    }

    // grad
    pub fn grad(&self) -> Arrayy {
        self.node.lock().unwrap().grad.clone()
    }

    // shape
    pub fn shape(&self) -> Vec<usize> {
        self.node.lock().unwrap().value.shape.clone()
    }

    // update
    // parent
    pub fn update_parent(&self, parent: Vec<NodeType>) {
        self.node.lock().as_mut().unwrap().parent = parent;
    }

    // label
    pub fn update_label(&self, label: Option<BackwardLabel>) {
        self.node.lock().as_mut().unwrap().label = label;
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let string = format!("{}", self.node.lock().unwrap().value);
        f.write_str(&string)
    }
}

// impl Mul for Tensor {
//     type Output = Self;
//     fn mul(self, rhs: Self) -> Self::Output {
//         let output = self.value() * rhs.value();
//         Tensor::from_arrayy(output)
//     }
// }
