use std::{ fmt::Display, ops::Range, sync::{ Arc, Mutex, RwLock } };

use rand::random;

use crate::{
    rotta_rs_module::{ arrayy::{ Arrayy, RecFlatten }, BackwardLabel, Node, NodeType },
    TensorRange,
    TensorRef,
};

#[derive(Debug, Clone)]
pub struct Tensor {
    pub node: NodeType,
}

impl Tensor {
    // initialization
    pub fn new<T: RecFlatten>(arr: T) -> Tensor {
        let (shape, vector) = arr.rec_flatten();
        let arr = Arrayy::from_vector(shape, vector);
        let node = Arc::new(RwLock::new(Node::new(arr)));

        Tensor {
            node,
        }
    }

    pub fn from_arrayy(array: Arrayy) -> Tensor {
        let node = Node::new(array);
        let node = Arc::new(RwLock::new(node));

        let tensor = Tensor {
            node,
        };

        tensor
    }

    pub fn from_vector(shape: Vec<usize>, vector: Vec<f64>) -> Tensor {
        let array = Arrayy::from_vector(shape, vector);

        let node = Node::new(array);
        let node = Arc::new(RwLock::new(node));

        let tensor = Tensor {
            node,
        };

        tensor
    }

    pub fn from_element(shape: Vec<usize>, element: f64) -> Tensor {
        let array = Arrayy::arrayy_from_element(shape, element);

        let node = Node::new(array);
        let node = Arc::new(RwLock::new(node));

        let tensor = Tensor {
            node,
        };

        tensor
    }

    pub fn from_shape_fn<T: FnMut() -> f64>(shape: Vec<usize>, f: T) -> Tensor {
        let array = Arrayy::arrayy_from_shape_fn(shape, f);

        let node = Node::new(array);
        let node = Arc::new(RwLock::new(node));

        let tensor = Tensor {
            node,
        };

        tensor
    }

    pub fn rand(shape: Vec<usize>) -> Tensor {
        Tensor::from_arrayy(Arrayy::arrayy_from_shape_fn(shape, || random::<f64>()))
    }

    pub fn arange(range: Range<usize>) -> TensorRange {
        TensorRange::init(range)
    }

    pub fn zeros(shape: Vec<usize>) -> Tensor {
        Tensor::from_arrayy(Arrayy::zeros(shape))
    }

    // get
    // value
    pub fn value(&self) -> Arrayy {
        self.node.read().unwrap().value.clone()
    }

    // requires gradient
    pub fn requires_grad(&self) -> bool {
        self.node.read().unwrap().requires_grad
    }

    // grad
    pub fn grad(&self) -> Arrayy {
        self.node.read().unwrap().grad.clone()
    }

    // shape
    pub fn shape(&self) -> Vec<usize> {
        self.node.read().unwrap().value.shape.clone()
    }

    // update
    // parent
    pub fn update_parent(&self, parent: Vec<NodeType>) {
        self.node.write().as_mut().unwrap().parent = parent;
    }

    // label
    pub fn update_label(&self, label: Option<BackwardLabel>) {
        self.node.write().as_mut().unwrap().label = label;
    }

    pub fn update_parent_label(&self, parent: Vec<NodeType>, label: Option<BackwardLabel>) {
        let mut node = self.node.write().unwrap();
        node.parent = parent;
        node.label = label;
    }

    // requires grad
    pub fn set_requires_grad(&self, stat: bool) {
        self.node.write().unwrap().requires_grad = stat;
    }

    // auto zero grad
    pub fn set_auto_zero_grad(&self, stat: bool) {
        self.node.write().unwrap().auto_zero_grad = stat;
    }

    // tensor_ref
    // pub fn get_tensor_ref(&self){
    //     TensorRef{
    //         value: &self.node.read().unwrap().value
    //     }
    // }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let string = format!("{}", self.node.read().unwrap().value);
        f.write_str(&string)
    }
}
