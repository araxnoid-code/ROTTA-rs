use std::{ sync::{ Arc, Mutex }, thread };

use rayon::iter::{ IntoParallelRefIterator, ParallelIterator };

use crate::rotta_rs::*;

// function
pub fn add_arr(arr_a: &Arrayy, arr_b: &Arrayy) -> Arrayy {
    let arr_a_s = arr_a.shape.clone();
    let arr_b_s = arr_b.shape.clone();

    if arr_a_s == arr_b_s {
        //
        let vector = arr_a.value
            .iter()
            .enumerate()
            .map(|(i, v)| { v + arr_b.value[i] })
            .collect::<Vec<f64>>();

        let array = Arrayy::from_vector(arr_a_s, vector);
        return array;
    } else if arr_a_s.multiple_sum() == 1 || arr_b_s.multiple_sum() == 1 {
        // skalar
        if arr_a_s.multiple_sum() == 1 {
            let skalar = arr_a.value[0];

            let vector = arr_b.value
                .iter()
                .map(|v| skalar + *v)
                .collect::<Vec<f64>>();

            Arrayy::from_vector(arr_b_s.clone(), vector)
        } else {
            let skalar = arr_b.value[0];

            let vector = arr_a.value
                .iter()
                .map(|v| *v + skalar)
                .collect::<Vec<f64>>();

            Arrayy::from_vector(arr_a_s.clone(), vector)
        }
    } else {
        // broadcasting
        let b_shape = broadcast_concat(arr_a, arr_b);

        let mut a = if let Ok(arr) = broadcasting(arr_a, b_shape.clone()) {
            arr
        } else {
            (*arr_a).clone()
        };
        let b = if let Ok(arr) = broadcasting(arr_b, b_shape) { arr } else { (*arr_b).clone() };

        for i in 0..a.value.len() {
            a.value[i] += b.value[i];
        }
        a
    }
}

pub fn add_arr_slice(arr_a: (&[f64], &[usize]), arr_b: (&[f64], &[usize])) {
    let (arr_a, shape_a) = arr_a;
    let (arr_b, shape_b) = arr_b;

    if shape_a == shape_b {
        // same shape

        let vector = arr_a
            .iter()
            .enumerate()
            .map(|(i, a)| {
                // add
                a + arr_b[i]
            })
            .collect::<Vec<f64>>();
        (vector, shape_a);
    } else if shape_a.multiple_sum() == 1 || shape_b.multiple_sum() == 1 {
        // skalar
        if shape_a.multiple_sum() == 1 {
            let skalar = arr_a[0];

            let vector = arr_b
                .iter()
                .map(|v| skalar + *v)
                .collect::<Vec<f64>>();

            (vector, shape_b);
        } else {
            let skalar = arr_b[0];

            let vector = arr_a
                .iter()
                .map(|v| *v + skalar)
                .collect::<Vec<f64>>();

            (vector, shape_a);
        }
    } else {
        // broadcasting
    }
}

pub fn par_add_arr(arr_a: &Arrayy, arr_b: &Arrayy) -> Arrayy {
    let shape_a = arr_a.shape.as_slice();
    let shape_b = arr_b.shape.as_slice();

    // if shape_a == shape_b {
    let batch = shape_a[0];
    let length = (&shape_a[1..]).multiple_sum();

    let mapping = (0..batch)
        .map(|i| {
            let start = i * length;
            let stop = start + length;

            let arr_a = &arr_a.value[start..stop];

            let arr_b = &arr_b.value[start..stop];

            (arr_a, arr_b)
        })
        .collect::<Vec<(&[f64], &[f64])>>();

    let result = mapping
        .par_iter()
        .map(|(a, b)| {
            a.into_iter()
                .enumerate()
                .map(|(i, a)| { a + b[i] })
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>();

    let vector = result.concat();
    Arrayy::from_vector(shape_a.to_vec(), vector)
    // }
}
