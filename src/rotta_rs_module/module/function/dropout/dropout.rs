use std::sync::{ Arc, Mutex };

use rand::{ random, SeedableRng };
use rand_chacha::ChaCha8Rng;
use rand_distr::{ Bernoulli, Distribution };

use crate::rotta_rs_module::{ arrayy::Arrayy, Tensor };

pub struct Dropout {
    pub bernoulli: Bernoulli,
    pub p: f64,
    pub rng: ChaCha8Rng,
    pub eval: Arc<Mutex<bool>>,
}

impl Dropout {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        if !self.eval_status() {
            let r = self.r(x.shape());
            r.set_requires_grad(false);

            x * &r
        } else {
            (*x).clone()
        }
    }

    fn r(&self, shape: Vec<usize>) -> Tensor {
        let mut rng = ChaCha8Rng::seed_from_u64(random());
        let arr = Arrayy::arrayy_from_shape_fn(shape, || {
            let prob = self.bernoulli.sample(&mut rng);
            if prob {
                0.0
            } else {
                1.0 / (1.0 - self.p)
            }
        });
        Tensor::from_arrayy(arr)
    }

    pub fn eval_status(&self) -> bool {
        *self.eval.lock().unwrap()
    }

    pub fn eval(&self) {
        *self.eval.lock().unwrap() = true;
    }

    pub fn train(&self) {
        *self.eval.lock().unwrap() = false;
    }
}
