use crate::rotta_rs::*;

pub fn sum_axis_arr(arr: &Arrayy, d: i32) -> Arrayy {
    let mut shape = arr.shape.clone();

    // sum axis negative indexing
    let d = if d < 0 { ((shape.len() as i32) + d) as usize } else { d as usize };

    if d >= shape.len() {
        panic!("array is {} dimension but will sum in dimension {}", shape.len(), d);
    }

    let len_items = (&arr.shape[d + 1..]).multiple_sum();

    let mut vector = vec![];
    let mut vec = vec![];
    let count_loop = arr.value.len() / len_items;
    let mut counter = 1;
    for i in 0..count_loop {
        let start = i * len_items;
        let stop = start + len_items;
        let slice = &arr.value[start..stop];

        if vector.is_empty() {
            vector = slice.to_vec();
        } else {
            let sum = vector
                .iter()
                .enumerate()
                .map(|(i, v)| { v + slice[i] })
                .collect::<Vec<f64>>();
            vector = sum;
        }

        if counter >= arr.shape[d] {
            counter = 1;
            vec.push(vector.clone());
            vector = vec![];
        } else {
            counter += 1;
        }
    }

    shape.remove(d);
    Arrayy::from_vector(shape, vec.concat())
}
