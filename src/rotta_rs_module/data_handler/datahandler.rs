use std::{ sync::{ Arc, RwLock }, thread };

use rand::{ seq::SliceRandom, SeedableRng };
use rand_chacha::ChaCha8Rng;

use crate::{ arrayy::ArrSlice, ParDataHandler, Dataset, Tensor };

pub struct DataHandler {
    rng: ChaCha8Rng,
    idx: usize,
    batch: usize,
    dataset: Arc<RwLock<Vec<(Tensor, Tensor)>>>,

    //
    batch_: usize,

    // par_sampler
    par_idx: usize,
    par_num: usize,
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
            dataset: Arc::new(RwLock::new(combine)),
            batch_: 0,

            // par_sampler
            par_idx: 0,
            par_num: 0,
        }
    }

    // get
    // pub fn get(&self, idx: usize) -> (Tensor, Tensor) {
    //     // self.dataset.get(idx)
    //     *self.dataset.read().unwrap().get(idx).unwrap()
    // }

    pub fn len(&self) -> usize {
        self.dataset.read().unwrap().len()
    }

    // operations
    pub fn shuffle(&mut self) {
        self.dataset.write().unwrap().shuffle(&mut self.rng);
    }

    // update
    pub fn set_seed(&mut self, seed: u64) {
        self.rng = ChaCha8Rng::seed_from_u64(seed);
    }

    pub fn batch(&mut self, batch: usize) {
        self.batch = batch;
    }

    // multithread
    pub fn par_by_sample<F: CloneableFn<M>, M: ParDataHandler + 'static + Send + Sync + Clone>(
        &mut self,
        model: M,
        par_num: usize,
        f: F
    ) -> (Tensor, M) {
        let len = self.len();

        let mut loss = Tensor::new([0.0]);
        let model_arc = Arc::new(model.clone());

        let par_num = if self.par_num == 0 {
            if par_num > len {
                panic!("DATAHANDLER ERROR: par_unit must lower than length of dataset");
            }
            self.par_num = par_num;
            par_num
        } else {
            self.par_num
        };

        let mut handles = vec![];

        for sample_idx in self.par_idx..self.par_idx + par_num {
            if sample_idx >= len {
                self.par_idx = 0;
                break;
            }

            let f = f.clone_box();
            let model_arc = model_arc.clone();
            let dataset = self.dataset.clone();

            let handle = thread::spawn(move || {
                let _data = dataset.read().unwrap();
                let data = _data.get(sample_idx).unwrap();
                f(&data, &*model_arc)
            });
            handles.push(handle);
        }

        let mut iteration = 0.0;
        for handle in handles {
            let _loss = handle.join().unwrap();
            loss = &loss + &_loss;
            iteration += 1.0;
        }

        if self.par_idx < len - 1 {
            if self.par_idx + self.par_num >= len {
                self.par_idx = 0;
            }
            self.par_idx += self.par_num;
        } else {
            self.par_idx = 0;
        }

        (&loss / (iteration as f32), model)
    }
}

// iterator
impl<'a> Iterator for &'a mut DataHandler {
    type Item = (Tensor, Tensor);
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.len() {
            self.idx = 0;
            self.idx = 0;
            return None;
        }

        let sample_lock = self.dataset.read().unwrap();
        let sample = sample_lock.get(self.idx).unwrap();
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

//
pub trait CloneableFn<M>
    : FnOnce(&(Tensor, Tensor), &M) -> Tensor + 'static + Send
    where M: ParDataHandler + 'static
{
    fn clone_box(&self) -> Box<dyn CloneableFn<M>>;
}

impl<
    M: ParDataHandler + 'static,
    T: FnOnce(&(Tensor, Tensor), &M) -> Tensor + 'static + Send + Clone
> CloneableFn<M> for T {
    fn clone_box(&self) -> Box<dyn CloneableFn<M>> {
        Box::new(self.clone())
    }
}

impl<M: ParDataHandler + 'static> Clone for Box<dyn CloneableFn<M>> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
