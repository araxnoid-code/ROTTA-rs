use std::sync::{ Arc, Mutex };

use rand::{ rngs::StdRng, SeedableRng };

use crate::rotta_rs::{ NodeType, WeightInitialization };

pub struct Module {
    pub parameters: Arc<Mutex<Vec<NodeType>>>,
    pub initialization: WeightInitialization,

    // rng
    pub rng: StdRng,
}

impl Module {
    // initialization
    pub fn init() -> Module {
        Module {
            parameters: Arc::new(Mutex::new(Vec::new())),
            initialization: WeightInitialization::Glorot,

            // rng
            rng: StdRng::seed_from_u64(42),
        }
    }

    // parameters
    pub fn parameters(&self) -> Arc<Mutex<Vec<Arc<Mutex<crate::rotta_rs::Node>>>>> {
        self.parameters.clone()
    }

    // update seed
    pub fn update_seed(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
    }
}
