use rayon::iter::{ IntoParallelRefIterator, ParallelIterator };

use crate::rotta_rs_module::{
    arrayy::{ add_arr_slice, broadcast_shape_slice, broadcasting_arr_slice, Arrayy, MultipleSum },
};

// function
pub fn add_arr(arr_a: &Arrayy, arr_b: &Arrayy) -> Arrayy {
    let (vector, shape) = add_arr_slice((&arr_a.value, &arr_a.shape), (&arr_b.value, &arr_b.shape));
    Arrayy::from_vector(shape, vector)
}

pub fn par_add_arr(arr_a: &Arrayy, arr_b: &Arrayy) -> Arrayy {
    let shape_a = arr_a.shape.as_slice();
    let shape_b = arr_b.shape.as_slice();

    if shape_a == shape_b {
        let batch = shape_a[0];
        let length = (&shape_a[1..]).multiple_sum();

        let result = vec![0;batch]
            .par_iter()
            .map(|i| {
                let start = i * length;
                let stop = start + length;

                let arr_a = &arr_a.value[start..stop];
                let arr_b = &arr_b.value[start..stop];

                arr_a
                    .into_iter()
                    .enumerate()
                    .map(|(i, a)| { a + arr_b[i] })
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>();

        let vector = result.concat();
        Arrayy::from_vector(shape_a.to_vec(), vector)
    } else if shape_a.multiple_sum() == 1 || shape_b.multiple_sum() == 1 {
        // skalar
        if shape_a.multiple_sum() == 1 {
            let skalar = arr_a.value[0];
            let batch = shape_b[0];
            let length = (&shape_b[1..]).multiple_sum();

            let result = vec![0;batch]
                .par_iter()
                .map(|i| {
                    let start = i * length;
                    let stop = start + length;

                    (&arr_b.value[start..stop])
                        .into_iter()
                        .map(|b| skalar + b)
                        .collect::<Vec<f32>>()
                })
                .collect::<Vec<Vec<f32>>>();

            let vector = result.concat();
            Arrayy::from_vector(shape_b.to_vec(), vector)
        } else {
            let skalar = arr_b.value[0];
            let batch = shape_a[0];
            let length = (&shape_a[1..]).multiple_sum();

            let result = vec![0;batch]
                .par_iter()
                .map(|i| {
                    let start = i * length;
                    let stop = start + length;

                    (&arr_a.value[start..stop])
                        .into_iter()
                        .map(|b| b + skalar)
                        .collect::<Vec<f32>>()
                })
                .collect::<Vec<Vec<f32>>>();

            let vector = result.concat();
            Arrayy::from_vector(shape_a.to_vec(), vector)
        }
    } else {
        // broadcasting
        let broadcast_shape = broadcast_shape_slice(shape_a, shape_b).unwrap();

        let (arr_a, shape_a) = broadcasting_arr_slice(
            (&arr_a.value, &arr_a.shape),
            &broadcast_shape
        );
        let (arr_b, _) = broadcasting_arr_slice((&arr_b.value, &arr_b.shape), &broadcast_shape);

        let batch = shape_a[0];
        let length = (&shape_a[1..]).multiple_sum();

        let result = vec![0;batch]
            .par_iter()
            .map(|i| {
                let start = i * length;
                let stop = start + length;

                let arr_a = &arr_a[start..stop];
                let arr_b = &arr_b[start..stop];

                arr_a
                    .into_iter()
                    .enumerate()
                    .map(|(i, a)| { a + arr_b[i] })
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>();

        let vector = result.concat();
        Arrayy::from_vector(shape_a.to_vec(), vector)
    }
}
