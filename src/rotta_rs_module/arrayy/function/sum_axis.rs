use crate::rotta_rs_module::arrayy::*;

pub fn sum_axis_arr(arr: &Arrayy, dims: &[i32]) -> Arrayy {
    let mut dims = dims
        .into_iter()
        .enumerate()
        .map(|(_, d)| {
            if *d >= 0 { *d as usize } else { ((arr.shape.len() as i32) + d) as usize }
        })
        .collect::<Vec<usize>>();
    dims.sort();

    let mut value = vec![];
    let mut _shape = arr.shape.clone();
    for d in &dims {
        let shape = &_shape;
        // sum axis negative indexing

        if *d >= shape.len() {
            panic!("array is {} dimension but will sum in dimension {}", shape.len(), d);
        }

        let len_items = (&shape[d + 1..]).multiple_sum();
        let count_loop = (&shape[..d + 1]).multiple_sum();
        let mut vec = vec![];
        let mut counter = 1;
        let mut vector = vec![];

        for i in 0..count_loop {
            let start = i * len_items;
            let stop = start + len_items;
            let slice = if value.is_empty() {
                &arr.value[start..stop]
            } else {
                // dbg!(&value);

                &value[start..stop]
            };

            if vector.is_empty() {
                vector.extend_from_slice(&slice);
            } else {
                let sum = vector
                    .iter()
                    .enumerate()
                    .map(|(i, v)| { v + slice[i] })
                    .collect::<Vec<f64>>();
                vector = sum;
            }

            if counter >= shape[*d] {
                counter = 1;

                vec.extend_from_slice(vector.as_slice());
                vector.clear();
            } else {
                counter += 1;
            }
        }

        _shape[*d] = 1;
        value = vec;
    }

    let mut new_shape = _shape;
    let mut i = 0;
    for d in dims {
        new_shape.remove(d - i);
        i += 1;
    }

    Arrayy::from_vector(new_shape, value)
}

pub fn sum_axis_keep_dim_arr(arr: &Arrayy, dims: &[i32]) -> Arrayy {
    let mut dims = dims
        .into_iter()
        .enumerate()
        .map(|(_, d)| {
            if *d >= 0 { *d as usize } else { ((arr.shape.len() as i32) + d) as usize }
        })
        .collect::<Vec<usize>>();
    dims.sort();

    let mut value = vec![];
    let mut _shape = arr.shape.clone();
    for d in &dims {
        let shape = &_shape;
        // sum axis negative indexing

        if *d >= shape.len() {
            panic!("array is {} dimension but will sum in dimension {}", shape.len(), d);
        }

        let len_items = (&shape[d + 1..]).multiple_sum();
        let count_loop = (&shape[..d + 1]).multiple_sum();
        let mut vec = vec![];
        let mut counter = 1;
        let mut vector = vec![];

        for i in 0..count_loop {
            let start = i * len_items;
            let stop = start + len_items;
            let slice = if value.is_empty() {
                &arr.value[start..stop]
            } else {
                // dbg!(&value);

                &value[start..stop]
            };

            if vector.is_empty() {
                vector.extend_from_slice(&slice);
            } else {
                let sum = vector
                    .iter()
                    .enumerate()
                    .map(|(i, v)| { v + slice[i] })
                    .collect::<Vec<f64>>();
                vector = sum;
            }

            if counter >= shape[*d] {
                counter = 1;

                vec.extend_from_slice(vector.as_slice());
                vector.clear();
            } else {
                counter += 1;
            }
        }

        _shape[*d] = 1;
        value = vec;
    }

    Arrayy::from_vector(_shape, value)
}
