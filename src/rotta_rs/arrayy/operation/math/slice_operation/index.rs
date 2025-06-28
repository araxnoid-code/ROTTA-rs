use crate::rotta_rs::arrayy::*;

pub fn slice_index(slice: &[f64], shape: &[usize], index: &[usize]) -> f64 {
    // let index = negative_indexing(&self.shape, index).unwrap();

    let mut out = 0;
    let mut count = 1;

    let mut new_shape = shape[index.len()..].to_vec();
    if new_shape.is_empty() {
        new_shape.push(1);
    }

    for index in index {
        let multiple = (&shape[count..]).multiple_sum();
        let pointing = index * multiple;
        out += pointing;

        count += 1;
    }

    let value = slice[out];

    value
}
