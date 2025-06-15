use crate::rotta_rs::*;

pub fn t_2d(arr: &Arrayy) -> Arrayy {
    let mut shape = arr.shape.clone();

    let mut vector = Vec::new();
    for row in 0..*shape.first().unwrap() {
        for coll in 0..*shape.last().unwrap() {
            vector.push(arr.index(vec![coll, row].as_slice()));
        }
    }

    shape.reverse();
    Arrayy::from_vector(shape, vector)
}
