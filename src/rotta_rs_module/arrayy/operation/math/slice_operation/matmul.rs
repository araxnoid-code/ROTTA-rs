use crate::rotta_rs_module::arrayy::*;

pub fn matmul_2d_slice(
    arr_a: (&[f64], &[usize]),
    arr_b: (&[f64], &[usize])
) -> (Vec<f64>, Vec<usize>) {
    let (arr_a, shape_a) = arr_a;
    let (arr_b, shape_b) = arr_b;

    if shape_a.len() != 2 || shape_b.len() != 2 {
        panic!("error can't matmul");
    }

    let m = shape_a[shape_a.len() - 2];
    let o = shape_b.last().unwrap();
    let n = shape_a.last().unwrap();

    let shape = vec![m, *o];
    let mut vector = Vec::with_capacity(shape.multiple_sum());

    for row in 0..m {
        for coll in 0..*o {
            let mut sum = 0.0;
            for i in 0..*n {
                sum =
                    sum +
                    slice_index(arr_a, shape_a, [row, i].as_slice()) *
                        slice_index(arr_b, shape_b, [i, coll].as_slice());
            }
            vector.push(sum);
        }
    }

    (vector, shape)
}

pub fn matmul_nd_slice(
    arr_a: (&[f64], &[usize]),
    arr_b: (&[f64], &[usize])
) -> (Vec<f64>, Vec<usize>) {
    let (arr_a, shape_a) = arr_a;
    let (arr_b, shape_b) = arr_b;

    if shape_a.len() == 1 || shape_b.len() == 1 {
        panic!("can't matmul array cause, {:?} will matmul with {:?}", shape_a, shape_b);
    }

    if shape_a.len() == 2 && shape_b.len() == 2 {
        matmul_2d_slice((arr_a, shape_a), (arr_b, shape_b))
    } else {
        let shape_a_2d = &shape_a[shape_a.len() - 2..];
        let length_a = shape_a_2d.multiple_sum();

        let shape_b_2d = &shape_b[shape_b.len() - 2..];
        let length_b = shape_b_2d.multiple_sum();

        let mut shape = shape_a.to_vec();
        let len = shape.len();
        shape[len - 1] = *shape_b.last().unwrap();
        shape[len - 2] = shape_a[shape_a.len() - 2];
        let mut vector = Vec::with_capacity(shape.as_slice().multiple_sum());
        let looping = (&shape_a[..shape_a.len() - 2]).multiple_sum();

        for i in 0..looping {
            let start_a = i * length_a;
            let stop_a = start_a + length_a;
            let slice_a = &arr_a[start_a..stop_a];

            let start_b = i * length_b;
            let stop_b = start_b + length_b;
            let slice_b = &arr_b[start_b..stop_b];

            let (vec, _) = matmul_2d_slice((slice_a, shape_a_2d), (slice_b, shape_b_2d));
            vector.extend(vec.into_iter());
        }

        (vector, shape)
    }
}
