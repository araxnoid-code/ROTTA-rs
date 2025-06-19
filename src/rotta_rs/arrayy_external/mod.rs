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
    fn rec_flatten(&self) -> (Vec<usize>, Vec<f64>);

    fn recursive(&self, idx: usize, shape: &mut Vec<usize>, output: &mut Vec<f64>);
}

impl RecFlatten for f64 {
    fn rec_flatten(&self) -> (Vec<usize>, Vec<f64>) {
        (Vec::new(), Vec::new())
    }

    fn recursive(&self, _: usize, _: &mut Vec<usize>, output: &mut Vec<f64>) {
        output.push(*self);
    }
}

// impl<T: Debug + RecFlatten> RecFlatten for Vec<T> {
//     fn rec_flatten(&self) -> Vec<f64> {
//         let mut output: Vec<f64> = Vec::new();

//         self.recursive(&mut output);
//         output
//     }

//     fn recursive(&self, output: &mut Vec<f64>) {
//         for item in self {
//             item.recursive(output);
//         }
//     }
// }

impl<T: Debug + RecFlatten, const N: usize> RecFlatten for [T; N] {
    fn rec_flatten(&self) -> (Vec<usize>, Vec<f64>) {
        let mut output: Vec<f64> = Vec::new();

        let mut shape: Vec<usize> = vec![];

        self.recursive(0, &mut shape, &mut output);

        (shape, output)
    }

    fn recursive(&self, idx: usize, shape: &mut Vec<usize>, output: &mut Vec<f64>) {
        if idx == 0 {
            shape.push(self.len());
        }

        for (idx_enum, item) in self.iter().enumerate() {
            let idx = if idx == 0 { idx_enum } else { 1 };

            item.recursive(idx, shape, output);
        }
    }
}
