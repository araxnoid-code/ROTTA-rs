use crate::rotta_rs::{
    abs,
    exp,
    index,
    index_replace,
    permute,
    powi,
    reshape,
    sign,
    slice,
    slice_replace,
    sum,
    sum_axis,
    to_shape as to_shape_tensor,
    transpose,
    ArrSlice,
    Tensor,
};

impl Tensor {
    pub fn t(&self) -> Tensor {
        transpose(self, (-1, -2))
    }

    pub fn transpose(&self, d: (i32, i32)) -> Tensor {
        transpose(self, d)
    }

    pub fn exp(&self) -> Tensor {
        exp(self)
    }

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

    pub fn sum_axis(&self, d: i32) -> Tensor {
        sum_axis(self, d)
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

    pub fn slice(&self, range: Vec<ArrSlice>) -> Tensor {
        slice(self, range)
    }

    pub fn slice_replcae(&self, range: Vec<ArrSlice>, replace: &Tensor) {
        slice_replace(self, range, replace);
    }

    pub fn to_shape(&self, to_shape: Vec<usize>) -> Tensor {
        to_shape_tensor(self, to_shape)
    }

    pub fn reshape(&self, re_shape: Vec<i32>) -> Tensor {
        reshape(self, re_shape)
    }

    pub fn sign(&self) -> Tensor {
        sign(self)
    }
}
