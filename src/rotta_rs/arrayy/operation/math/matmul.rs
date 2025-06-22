use crate::rotta_rs::*;

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
                    sum = sum + (arr_a.index(vec![row, i]) * arr_b.index(vec![i, coll])).value[0];
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
    if arr_a.shape.len() == 1 || arr_b.shape.len() == 1 {
        panic!("can't matmul array cause, {:?} will matmul with {:?}", arr_a.shape, arr_b.shape)
    } else if arr_a.shape.len() == 2 && arr_b.shape.len() == 2 {
        matmul_2d(arr_a, arr_b)
    } else {
        let shape_a = arr_a.shape.clone();
        let shape_b = arr_b.shape.clone();
        let mut index = vec![];
        let mut d = 0;

        let mut shape = shape_a.clone();
        let len = shape.len();
        shape[len - 1] = *shape_b.last().unwrap();
        shape[len - 2] = shape_a[shape_a.len() - 2];
        let mut vector = Vec::with_capacity(shape.as_slice().multiple_sum());

        loop {
            if (shape_a.len() as i32) - (d as i32) == 2 {
                // matmul operation
                let slice_range = index
                    .iter()
                    .map(|d| { ArrSlice(Some(*d), Some(d + 1)) })
                    .collect::<Vec<ArrSlice>>();

                let slice_a = slice(&arr_a, slice_range.clone()).squeeze();
                let slice_b = slice(&arr_b, slice_range).squeeze();

                let matmul = matmul_2d(&slice_a, &slice_b);
                vector.extend(matmul.value.into_iter());

                d -= 1;
            } else {
                if let None = index.get(d) {
                    index.push(0);
                    d += 1;
                } else {
                    index[d] += 1;

                    if index[d] >= shape_a[d] {
                        index.pop();

                        if d == 0 {
                            break;
                        } else {
                            d -= 1;
                        }
                    } else {
                        d += 1;
                    }
                }
            }
        }

        Arrayy::from_vector(shape, vector)
    }
}
