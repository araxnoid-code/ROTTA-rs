use crate::rotta_rs::{ abs, index, index_replace, permute, powi, sum, Tensor };

impl Tensor {
    pub fn powi(&self, n: i32) -> Tensor {
        powi(self, n)
    }

    pub fn sum(&self) -> Tensor {
        sum(self)
    }

    pub fn len(&self) -> usize {
        self.node.lock().unwrap().value.len()
    }

    pub fn abs(&self) -> Tensor {
        abs(self)
    }

    pub fn index(&self, idx: Vec<i32>) -> Tensor {
        index(self, idx)
    }

    pub fn index_replace(&self, index: Vec<i32>, replace: Tensor) {
        if !self.node.lock().unwrap().requires_grad {
            index_replace(self, index, replace);
        } else {
            panic!("{}", "can't change manualy a tensor if the tensor is requires_grad=true")
        }
    }

    pub fn permute(&self, order: Vec<usize>) -> Tensor {
        permute(self, order)
    }
}
