use crate::rotta_rs::{ broadcast_shape_slice, broadcasting_arr_slice, MultipleSum };

pub fn add_arr_slice(
    arr_a: (&[f64], &[usize]),
    arr_b: (&[f64], &[usize])
) -> (Vec<f64>, Vec<usize>) {
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
        (vector, shape_a.to_vec())
    } else if shape_a.multiple_sum() == 1 || shape_b.multiple_sum() == 1 {
        // skalar
        if shape_a.multiple_sum() == 1 {
            let skalar = arr_a[0];

            let vector = arr_b
                .iter()
                .map(|v| skalar + *v)
                .collect::<Vec<f64>>();

            (vector, shape_b.to_vec())
        } else {
            let skalar = arr_b[0];

            let vector = arr_a
                .iter()
                .map(|v| *v + skalar)
                .collect::<Vec<f64>>();

            (vector, shape_a.to_vec())
        }
    } else {
        // broadcasting
        let broadcast_shape = broadcast_shape_slice(shape_a, shape_b).unwrap();

        let (arr_a, _) = broadcasting_arr_slice((arr_a, shape_a), &broadcast_shape);
        let (arr_b, _) = broadcasting_arr_slice((arr_b, shape_b), &broadcast_shape);

        let vector = arr_a
            .iter()
            .enumerate()
            .map(|(i, a)| {
                // add
                a + arr_b[i]
            })
            .collect::<Vec<f64>>();
        (vector, broadcast_shape)
    }
}
