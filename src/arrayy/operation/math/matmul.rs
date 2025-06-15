use crate::{ broadcasting, broadcast_concat, matmul_broadcasting, Arrayy, MultipleSum };

// function
pub fn matmul_2d(arr_a: &Arrayy, arr_b: &Arrayy) -> Arrayy {
    let arr_a_s = arr_a.shape.clone();
    let arr_b_s = arr_b.shape.clone();
    if arr_a_s.len() == 2 && arr_b_s.len() == 2 {
        // 2x2 * 2x2
        let m = arr_a.shape[arr_a.shape.len() - 2];
        let o = arr_b.shape.last().unwrap();
        let n = arr_a.shape.last().unwrap();

        let mut vector = vec![];
        for row in 0..m {
            for coll in 0..*o {
                let mut sum = 0.0;
                for i in 0..*n {
                    sum += arr_a.index(&[row, i][..]) * arr_b.index(&[i, coll][..]);
                }
                vector.push(sum);
            }
        }

        let array = Arrayy::from_vector(vec![m, *o], vector);

        return array;
    } else {
        panic!("error can't matmul")
    }
}

pub fn matmul_nd(arr_a: &Arrayy, arr_b: &Arrayy) -> Arrayy {
    let arr_a_s = arr_a.shape.clone();
    let arr_b_s = arr_b.shape.clone();
    let mut output_vector = Vec::new();

    if arr_a_s == arr_b_s {
        matmul_recursive(arr_a.clone(), arr_b.clone(), &mut output_vector);
    } else {
        // broadcasting
        let broad_shape = broadcast_concat(arr_a, arr_b);
        // println!("{:?}", broad_shape);

        let arr_a = if let Ok(b) = matmul_broadcasting(arr_a, broad_shape.clone()) {
            b
        } else {
            arr_a.clone()
        };
        let arr_b = if let Ok(b) = broadcasting(arr_b, broad_shape) { b } else { arr_b.clone() };

        matmul_recursive(arr_a.clone(), arr_b.clone(), &mut output_vector);
    }

    let output = output_vector.concat();
    let mut shape = arr_a_s.clone();
    let len = shape.len();
    shape[len - 1] = *arr_b_s.last().unwrap();
    shape[len - 2] = arr_a_s[arr_a_s.len() - 2];

    let arr = Arrayy::from_vector(shape, output);
    // arr_a.clone()
    arr
}

fn matmul_recursive(arr_a: Arrayy, arr_b: Arrayy, output_vector: &mut Vec<Vec<f64>>) {
    if arr_a.shape.len() == 2 {
        let arr = matmul_2d(&arr_a, &arr_b);
        output_vector.push(arr.value);
    } else {
        for d in 0..*arr_a.shape.first().unwrap() {
            // a
            let a_shape = &arr_a.shape[1..];
            let a_items = a_shape.multiple_sum();
            let start = d * a_items;
            let stop = start + a_items;
            let vector = arr_a.value[start..stop].to_vec();
            let arr_a = Arrayy::from_vector(a_shape.to_vec(), vector);

            // b
            let b_shape = &arr_b.shape[1..];
            let b_items = b_shape.multiple_sum();
            let start = d * b_items;
            let stop = start + b_items;
            let vector = arr_b.value[start..stop].to_vec();
            let arr_b = Arrayy::from_vector(b_shape.to_vec(), vector);

            matmul_recursive(arr_a, arr_b, output_vector);
        }
    }
}
