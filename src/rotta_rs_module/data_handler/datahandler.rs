use rand::{ seq::SliceRandom, SeedableRng };
use rand_chacha::ChaCha8Rng;

use crate::{ arrayy::ArrSlice, Dataset, Tensor };

pub struct DataHandler {
    rng: ChaCha8Rng,
    idx: usize,
    batch: usize,
    dataset: Vec<(Tensor, Tensor)>,

    //
    batch_: usize,
}

impl DataHandler {
    // initialization
    pub fn init<T: Dataset>(dataset: T) -> DataHandler {
        let mut combine = Vec::with_capacity(dataset.len());
        for i in 0..dataset.len() {
            combine.push(dataset.get(i));
        }

        DataHandler {
            rng: ChaCha8Rng::seed_from_u64(42),
            idx: 0,
            batch: 128,
            dataset: combine,

            batch_: 0,
        }
    }

    // get
    pub fn get(&self, idx: usize) -> Option<&(Tensor, Tensor)> {
        self.dataset.get(idx)
    }

    pub fn len(&self) -> usize {
        self.dataset.len()
    }

    // operations
    pub fn shuffle(&mut self) {
        self.dataset.shuffle(&mut self.rng);
    }

    // update
    pub fn set_seed(&mut self, seed: u64) {
        self.rng = ChaCha8Rng::seed_from_u64(seed);
    }

    pub fn batch(&mut self, batch: usize) {
        self.batch = batch;
    }
}

// iterator
impl<'a> Iterator for &'a mut DataHandler {
    type Item = (Tensor, Tensor);
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.dataset.len() {
            self.idx = 0;
            self.idx = 0;
            return None;
        }

        let sample = self.dataset.get(self.idx).unwrap();
        let sample_batch = sample.0.shape()[0];
        let start = self.batch_ as i32;

        let length = if self.batch_ + self.batch <= sample_batch {
            self.batch_ += self.batch;
            Some(start + (self.batch as i32))
        } else {
            self.batch_ += sample_batch;
            None
        };

        if self.batch_ >= sample_batch {
            self.idx += 1;
            self.batch_ = 0;
        }

        let input = sample.0.slice(&[ArrSlice(Some(start), length)]);
        let label = sample.1.slice(&[ArrSlice(Some(start), length)]);

        Some((input, label))
    }
}
