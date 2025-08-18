use std::{ sync::{ Arc, Mutex } };

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::{ rotta_rs_module::WeightInitialization, ShareTensor };

pub struct Module {
    pub parameters: Arc<Mutex<Vec<ShareTensor>>>,
    pub initialization: WeightInitialization,
    pub eval_handlers: Vec<Arc<Mutex<bool>>>,

    // rng
    pub rng: ChaCha8Rng,
}

impl Module {
    // initialization
    pub fn init() -> Module {
        Module {
            parameters: Arc::new(Mutex::new(Vec::new())),
            initialization: WeightInitialization::He,
            eval_handlers: vec![],

            // rng
            rng: ChaCha8Rng::seed_from_u64(42),
        }
    }

    // parameters
    pub fn parameters(&self) -> Arc<Mutex<Vec<ShareTensor>>> {
        self.parameters.clone()
    }

    // eval & training
    pub fn train(&self) {
        for p in &self.eval_handlers {
            *p.lock().unwrap() = false;
        }
    }

    pub fn eval(&self) {
        for p in &self.eval_handlers {
            *p.lock().unwrap() = true;
        }
    }

    // update seed
    pub fn update_seed(&mut self, seed: u64) {
        self.rng = ChaCha8Rng::seed_from_u64(seed);
    }

    pub fn update_initialization(&mut self, initialization: WeightInitialization) {
        self.initialization = initialization;
    }
}
