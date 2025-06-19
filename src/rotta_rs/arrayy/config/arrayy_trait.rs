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

// i32 and f64
// pub trait ArrayyType {}

// impl ArrayyType for i32 {}

// impl ArrayyType for f64 {}

// impl Debug for Box<ArrayyType> {
//     // fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {}
// }
