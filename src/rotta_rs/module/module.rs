use std::sync::{ Arc, Mutex };

use crate::rotta_rs::{ NodeType, WeightInitialization };

pub struct Module {
    pub parameters: Arc<Mutex<Vec<NodeType>>>,
    pub initialization: WeightInitialization,
}

impl Module {
    // initialization
    pub fn init() -> Module {
        Module {
            parameters: Arc::new(Mutex::new(Vec::new())),
            initialization: WeightInitialization::Random,
        }
    }

    // parameters
    pub fn parameters(&self) -> Arc<Mutex<Vec<Arc<Mutex<crate::rotta_rs::Node>>>>> {
        self.parameters.clone()
    }
}
