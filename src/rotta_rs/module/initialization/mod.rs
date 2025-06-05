mod random;
pub use random::*;

pub enum WeightInitialization {
    Random,
    He,
    Glorot,
}
