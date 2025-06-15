use crate::Arrayy;

pub fn permute(order: Vec<usize>, arr: &Arrayy) -> Arrayy {
    let shape = arr.shape.clone();

    let mut permute_shape = vec![];
    if order.len() != shape.len() {
        panic!(
            "permute list out of array shape, permute list is {:?}, but array shape is {:?}",
            order,
            shape
        );
    }
    for i in &order {
        if i >= &shape.len() {
            panic!(
                "permute list out of array shape, permute list is {:?}, but array shape is {:?}",
                order,
                shape
            );
        }
        permute_shape.push(shape[*i]);
    }

    let mut output = Arrayy::from_vector(permute_shape, vec![0.0; arr.value.len()]);
    let mut index: Vec<usize> = vec![];
    let mut current_d = 0;
    loop {
        if current_d >= shape.len() - 1 {
            // kolom
            if let None = index.get(current_d) {
                index.push(0);
            } else {
                // operation do here
                let mut permute_index = vec![];
                for i in &order {
                    permute_index.push(index[*i]);
                }
                output.index_mut(&permute_index, arr.index(&index));

                if index[current_d] < *shape.last().unwrap() - 1 {
                    index[current_d] += 1;
                } else {
                    index.pop();

                    if current_d == 0 {
                        break;
                    } else {
                        current_d -= 1;
                    }
                }
            }
        } else {
            if let None = index.get(current_d) {
                index.push(0);
                current_d += 1;
            } else {
                if index[current_d] < shape[current_d] - 1 {
                    index[current_d] += 1;
                    current_d += 1;
                } else {
                    index.pop();

                    if current_d == 0 {
                        break;
                    } else {
                        current_d -= 1;
                    }
                }
            }
        }
    }

    output
}
