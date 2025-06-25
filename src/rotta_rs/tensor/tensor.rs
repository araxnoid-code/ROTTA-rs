use std::{ fmt::Display, ops::Mul, sync::{ Arc, Mutex } };

use rand::random;

use crate::rotta_rs::{ Arrayy, BackwardLabel, MultipleSum, Node, NodeType, RecFlatten };

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

    pub fn rand(shape: Vec<usize>) -> Tensor {
        Tensor::from_arrayy(Arrayy::arrayy_from_shape_fn(shape, || random::<f64>()))
    }

    // get
    // value
    pub fn value(&self) -> Arrayy {
        self.node.lock().unwrap().value.clone()
    }

    // requires gradient
    pub fn requires_grad(&self) -> bool {
        self.node.lock().unwrap().requires_grad
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

    pub fn update_parent_label(&self, parent: Vec<NodeType>, label: Option<BackwardLabel>) {
        let mut node = self.node.lock().unwrap();
        node.parent = parent;
        node.label = label;
    }

    // requires grad
    pub fn set_requires_grad(&self, stat: bool) {
        self.node.lock().unwrap().requires_grad = stat;
    }

    // auto zero grad
    pub fn set_auto_zero_grad(&self, stat: bool) {
        self.node.lock().unwrap().auto_zero_grad = stat;
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
