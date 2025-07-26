mod base_operation;
use crate::rotta_rs_module::{ arrayy::* };

impl Arrayy {
    pub fn sum(&self) -> f64 {
        self.value.iter().sum::<f64>()
    }

    pub fn exp(&self) -> Arrayy {
        exp_arr(self)
    }

    pub fn dot(&self, rhs: &Arrayy) -> Arrayy {
        dot_arr(self, rhs)
    }

    pub fn matmul(&self, rhs: &Arrayy) -> Arrayy {
        // matmul_nd(self, rhs)
        let (vector, shape) = matmul_nd_slice(
            (self.value.as_slice(), self.shape.as_slice()),
            (rhs.value.as_slice(), rhs.shape.as_slice())
        );

        Arrayy::from_vector(shape, vector)
    }

    pub fn par_matmul(&self, rhs: &Arrayy) -> Arrayy {
        par_matmul_arr(self, rhs)
    }

    pub fn permute(&self, order: &Vec<usize>) -> Arrayy {
        permute_arr(&order, self)
    }

    pub fn ln(&self) -> Arrayy {
        ln_arr(self)
    }

    pub fn t(&self) -> Arrayy {
        transpose_arr(&self, (-1, -2))
    }

    pub fn transpose(&self, d: (i32, i32)) -> Arrayy {
        transpose_arr(self, d)
    }

    pub fn powi(&self, n: i32) -> Arrayy {
        powi_arr(self, n)
    }

    pub fn powf(&self, n: f64) -> Arrayy {
        powf_arr(self, n)
    }

    pub fn squeeze(&self) -> Arrayy {
        let shape = self.shape.clone()[1..].to_vec();

        Arrayy::from_vector(shape, self.value.clone())
    }

    pub fn len(&self) -> usize {
        self.value.len()
    }

    pub fn abs(&self) -> Arrayy {
        abs_arr(self)
    }

    pub fn sign(&self) -> Arrayy {
        sign_arr(self)
    }

    pub fn slice(&self, range: &[ArrSlice]) -> Arrayy {
        slice_arr(self, range)
    }

    pub fn slice_replace(&mut self, range: &[ArrSlice], replace: &Arrayy) {
        slice_replace_arr(self, range, replace);
    }

    pub fn to_shape(&self, to_shape: Vec<usize>) -> Arrayy {
        to_shape_arr(self, to_shape)
    }

    pub fn reshape(&self, reshape: Vec<i32>) -> Arrayy {
        reshape_arr(self, reshape)
    }

    pub fn sum_axis(&self, d: &[i32]) -> Arrayy {
        sum_axis_arr(self, d)
    }

    pub fn sum_axis_keep_dim(&self, d: &[i32]) -> Arrayy {
        sum_axis_keep_dim_arr(self, d)
    }

    pub fn mean(&self) -> Arrayy {
        mean_arr(self)
    }

    pub fn mean_axis(&self, d: &[i32]) -> Arrayy {
        mean_axis_arr(self, d)
    }

    pub fn mean_axis_keep_dim(&self, d: &[i32]) -> Arrayy {
        mean_axis_keep_dim_arr(self, d)
    }

    pub fn argmax(&self, dim: i32) -> Arrayy {
        argmax_arr(self, dim)
    }

    pub fn argmin(&self, dim: i32) -> Arrayy {
        argmin_arr(self, dim)
    }

    pub fn sin(&self) -> Arrayy {
        sin_arr(self)
    }

    pub fn cos(&self) -> Arrayy {
        cos_arr(self)
    }

    pub fn tan(&self) -> Arrayy {
        tan_arr(self)
    }
}
