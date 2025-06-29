use std::fmt::Debug;

pub trait MultipleSum {
    fn multiple_sum(&self) -> usize;
}

impl MultipleSum for &[usize] {
    fn multiple_sum(&self) -> usize {
        let mut result = 1;
        for item in self.iter() {
            result *= item;
        }

        result
    }
}

impl MultipleSum for Vec<usize> {
    fn multiple_sum(&self) -> usize {
        let mut result = 1;
        for item in self.iter() {
            result *= item;
        }

        result
    }
}
