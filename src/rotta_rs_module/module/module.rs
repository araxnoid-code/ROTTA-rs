use std::sync::{ Arc, Mutex };

use rand::{ rngs::StdRng, SeedableRng };

use crate::rotta_rs_module::{ NodeType, TrainEvalHandler, WeightInitialization };

pub struct Module {
    pub parameters: Arc<Mutex<Vec<NodeType>>>,
    pub initialization: WeightInitialization,
    pub eval_handlers: Vec<Arc<Mutex<bool>>>,

    // rng
    pub rng: StdRng,
}

impl Module {
    // initialization
    pub fn init() -> Module {
        Module {
            parameters: Arc::new(Mutex::new(Vec::new())),
            initialization: WeightInitialization::He,
            eval_handlers: vec![],

            // rng
            rng: StdRng::seed_from_u64(42),
        }
    }

    // parameters
    pub fn parameters(&self) -> Arc<Mutex<Vec<Arc<Mutex<crate::rotta_rs_module::Node>>>>> {
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
        self.rng = StdRng::seed_from_u64(seed);
    }

    pub fn update_initialization(&mut self, initialization: WeightInitialization) {
        self.initialization = initialization;
    }
}
