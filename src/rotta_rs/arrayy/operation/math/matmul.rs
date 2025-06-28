use rayon::prelude::*;

use crate::rotta_rs::arrayy::*;

// function
pub fn matmul_2d(arr_a: &Arrayy, arr_b: &Arrayy) -> Arrayy {
    let (vector, shape) = matmul_2d_slice(
        (arr_a.value.as_slice(), arr_a.shape.as_slice()),
        (arr_b.value.as_slice(), arr_b.shape.as_slice())
    );

    Arrayy::from_vector(shape, vector)
}

pub fn matmul_nd(arr_a: &Arrayy, arr_b: &Arrayy) -> Arrayy {
    let (vector, shape) = matmul_nd_slice(
        (arr_a.value.as_slice(), arr_a.shape.as_slice()),
        (arr_b.value.as_slice(), arr_b.shape.as_slice())
    );

    Arrayy::from_vector(shape, vector)
}

pub fn par_matmul_arr(arr_a: &Arrayy, arr_b: &Arrayy) -> Arrayy {
    // multithread
    let shape_a = &arr_a.shape[..];
    let shape_b = &arr_b.shape[..];

    let batch = shape_a[0];

    let mapping = (0..batch)
        .map(|i| {
            let mut to_shape = shape_a.to_vec();
            to_shape[0] = 1;
            let mut arr = arr_a.index(vec![i as i32]);
            arr.shape = to_shape;
            arr
        })
        .collect::<Vec<Arrayy>>();

    let result = mapping
        .par_iter()
        .map(|arr| {
            let (vec, _) = matmul_nd_slice(
                (arr.value.as_slice(), arr.shape.as_slice()),
                (arr_b.value.as_slice(), arr_b.shape.as_slice())
            );

            vec
        })
        .collect::<Vec<Vec<f64>>>();

    let vector = result.concat();

    let mut shape = shape_a.to_vec();
    let len = shape.len();
    shape[len - 1] = *shape_b.last().unwrap();
    shape[len - 2] = shape_a[shape_a.len() - 2];

    Arrayy::from_vector(shape, vector)
}
