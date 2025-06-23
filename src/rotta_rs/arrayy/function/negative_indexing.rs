pub fn negative_indexing(x: &Vec<usize>, idx: Vec<i32>) -> Result<Vec<usize>, String> {
    if idx.len() > x.len() {
        return Err(format!("indexing out of shape"));
    }

    let mut indexing = vec![];
    for (i, d) in idx.iter().enumerate() {
        if *d >= 0 {
            if *d >= (x[i] as i32) {
                // err
                return Err(format!("indexing out of shape"));
            }

            indexing.push(*d as usize);
        } else {
            let operate = (x.len() as i32) + d;
            if operate < 0 {
                // err
                return Err(format!("indexing out of shape"));
            }

            indexing.push(operate as usize);
        }
    }

    Ok(indexing)
}
