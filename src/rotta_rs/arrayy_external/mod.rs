mod array;
pub use array::*;

mod operation;
pub use operation::*;

mod config;
pub use config::*;

mod function;
pub use function::*;

use std::fmt::Debug;

pub trait RecFlatten {
    fn rec_flatten(&self) -> Vec<f64>;

    fn recursive(&self, output: &mut Vec<f64>);
}

impl RecFlatten for f64 {
    fn rec_flatten(&self) -> Vec<f64> {
        Vec::new()
    }

    fn recursive(&self, output: &mut Vec<f64>) {
        output.push(*self);
    }
}

impl<T: Debug + RecFlatten> RecFlatten for Vec<T> {
    fn rec_flatten(&self) -> Vec<f64> {
        let mut output: Vec<f64> = Vec::new();

        self.recursive(&mut output);
        output
    }

    fn recursive(&self, output: &mut Vec<f64>) {
        for item in self {
            item.recursive(output);
        }
    }
}
