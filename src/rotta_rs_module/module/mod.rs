mod module;
pub use module::*;

mod function;
pub use function::*;

mod initialization;
pub use initialization::*;

mod saving;
pub use saving::*;

pub trait TrainEvalHandler {
    fn train(&mut self);
    fn eval(&mut self);
}
