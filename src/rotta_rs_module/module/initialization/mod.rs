mod random;
pub use random::*;

mod glorot;
pub use glorot::*;

mod he;
pub use he::*;

pub enum WeightInitialization {
    Random,
    He,
    Glorot,
}
