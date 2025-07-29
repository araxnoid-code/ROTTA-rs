use std::{ fmt::Display, sync::{ Arc, RwLock } };

use rand::random;
use uuid::Uuid;

use crate::{ arrayy::{ Arrayy, RecFlatten }, BackwardLabel };

#[derive(Clone, Debug)]
pub struct Tensor {
    pub id: u128,
    pub value: Arc<RwLock<Arrayy>>,
    pub grad: Arc<RwLock<Arrayy>>,
    pub parent: Vec<Arc<RwLock<Tensor>>>,
    pub label: Option<BackwardLabel>,
    // requires
    pub requires_grad: Arc<RwLock<bool>>,
    pub auto_zero_grad: Arc<RwLock<bool>>,
}

impl Tensor {
    // initialization
    pub fn new<T: RecFlatten>(arr: T) -> Tensor {
        let (shape, vector) = arr.rec_flatten();
        let arr = Arrayy::from_vector(shape, vector);

        Tensor {
            id: Uuid::new_v4().as_u128(),
            grad: Arc::new(RwLock::new(Arrayy::zeros(arr.shape.clone()))),
            value: Arc::new(RwLock::new(arr)),
            parent: Vec::new(),
            label: None,
            requires_grad: Arc::new(RwLock::new(true)),
            auto_zero_grad: Arc::new(RwLock::new(true)),
        }
    }

    pub fn from_arrayy(array: Arrayy) -> Tensor {
        Tensor {
            id: Uuid::new_v4().as_u128(),
            grad: Arc::new(RwLock::new(Arrayy::zeros(array.shape.clone()))),
            value: Arc::new(RwLock::new(array)),
            parent: Vec::new(),
            label: None,
            requires_grad: Arc::new(RwLock::new(true)),
            auto_zero_grad: Arc::new(RwLock::new(true)),
        }
    }

    pub fn from_vector(shape: Vec<usize>, vector: Vec<f64>) -> Tensor {
        let array = Arrayy::from_vector(shape, vector);

        Tensor {
            id: Uuid::new_v4().as_u128(),
            grad: Arc::new(RwLock::new(Arrayy::zeros(array.shape.clone()))),
            value: Arc::new(RwLock::new(array)),
            parent: Vec::new(),
            label: None,
            requires_grad: Arc::new(RwLock::new(true)),
            auto_zero_grad: Arc::new(RwLock::new(true)),
        }
    }

    pub fn from_element(shape: Vec<usize>, element: f64) -> Tensor {
        let array = Arrayy::arrayy_from_element(shape, element);

        Tensor {
            id: Uuid::new_v4().as_u128(),
            grad: Arc::new(RwLock::new(Arrayy::zeros(array.shape.clone()))),
            value: Arc::new(RwLock::new(array)),
            parent: Vec::new(),
            label: None,
            requires_grad: Arc::new(RwLock::new(true)),
            auto_zero_grad: Arc::new(RwLock::new(true)),
        }
    }

    pub fn from_shape_fn<T: FnMut() -> f64>(shape: Vec<usize>, f: T) -> Tensor {
        let array = Arrayy::arrayy_from_shape_fn(shape, f);

        Tensor {
            id: Uuid::new_v4().as_u128(),
            grad: Arc::new(RwLock::new(Arrayy::zeros(array.shape.clone()))),
            value: Arc::new(RwLock::new(array)),
            parent: Vec::new(),
            label: None,
            requires_grad: Arc::new(RwLock::new(true)),
            auto_zero_grad: Arc::new(RwLock::new(true)),
        }
    }

    pub fn rand(shape: Vec<usize>) -> Tensor {
        Tensor::from_arrayy(Arrayy::arrayy_from_shape_fn(shape, || random::<f64>()))
    }

    // pub fn arange(range: Ra<usize>) -> TensorRange {
    //     TensorRange::init(range)
    // }

    pub fn zeros(shape: Vec<usize>) -> Tensor {
        Tensor::from_arrayy(Arrayy::zeros(shape))
    }

    // get
    // value
    pub fn value(&self) -> Arrayy {
        self.value.read().unwrap().clone()
    }

    // requires gradient
    pub fn requires_grad(&self) -> bool {
        *self.requires_grad.read().unwrap()
    }

    // grad
    pub fn grad(&self) -> Arrayy {
        self.grad.read().unwrap().clone()
    }

    // shape
    pub fn shape(&self) -> Vec<usize> {
        self.value.read().unwrap().shape.clone()
    }

    // update
    // parent
    pub fn update_parent(&mut self, parent: Vec<Arc<RwLock<Tensor>>>) {
        self.parent = parent;
    }

    // label
    pub fn update_label(&mut self, label: Option<BackwardLabel>) {
        self.label = label;
    }

    pub fn update_parent_label(
        &mut self,
        parent: Vec<Arc<RwLock<Tensor>>>,
        label: Option<BackwardLabel>
    ) {
        self.update_parent(parent);
        self.update_label(label);
    }

    pub fn zeros_grad(&self) {
        let arr = Arrayy::zeros(self.shape());
        *self.grad.write().unwrap() = arr;
    }

    pub fn ones_grad(&self) {
        let arr = Arrayy::ones(self.shape());
        *self.grad.write().unwrap() = arr;
    }

    // requires grad
    pub fn set_requires_grad(&self, stat: bool) {
        *self.requires_grad.write().unwrap() = stat;
    }

    // auto zero grad
    pub fn set_auto_zero_grad(&self, stat: bool) {
        *self.auto_zero_grad.write().unwrap() = stat;
    }

    pub fn add_grad(&mut self, grad: Arrayy) {
        if self.requires_grad() {
            let mut grad_tensor = self.grad.write().unwrap();
            *grad_tensor = &*grad_tensor + grad;
        }
    }

    // shared_tensor
    pub fn shared_tensor(&self) -> Arc<Tensor> {
        Arc::new(self.clone())
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let string = format!("{}", self.value.read().unwrap());
        f.write_str(&string)
    }
}
