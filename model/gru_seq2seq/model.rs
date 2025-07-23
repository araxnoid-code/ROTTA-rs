pub struct MySeq2SeqModel {
    _model: Module,
    length: usize,
    // encoder
    embedding_encoder: Embedding,
    layer_norm_encoder: LayerNorm,
    gru_encoder: Gru,

    // decoder
    embedding_decoder: Embedding,
    layer_norm_decoder: LayerNorm,
    gru_decoder: Gru,
    linear_decoder: Linear,
}

impl MySeq2SeqModel {
    pub fn init(vocab_num: usize, hidden: usize, length: usize) -> MySeq2SeqModel {
        let mut model = Module::init();
        model.update_initialization(rotta_rs::WeightInitialization::Glorot);

        Self {
            // encoder
            embedding_encoder: model.embedding_init(vocab_num, hidden),
            layer_norm_encoder: model.layer_norm_init(&[hidden]),
            gru_encoder: model.gru_init(hidden),

            // decoder
            embedding_decoder: model.embedding_init(vocab_num, hidden),
            layer_norm_decoder: model.layer_norm_init(&[hidden]),
            gru_decoder: model.gru_init(hidden),
            linear_decoder: model.liniar_init(hidden, vocab_num),

            // model
            _model: model,
            length,
        }
    }

    pub fn encoder(&mut self, x: &Tensor) -> Option<Tensor> {
        let embedded = self.embedding_encoder.forward(&x.reshape(vec![self.length as i32]));
        let embedded = self.layer_norm_encoder.forward(&embedded);

        let mut _hidden = None;
        for i in 0..self.length {
            let x = embedded.index(vec![i as i32]).reshape(vec![1, -1]);
            let out = self.gru_encoder.forward(&x, _hidden);
            _hidden = Some(out);
        }

        _hidden
    }

    pub fn decoder(&mut self, context_vector: Option<Tensor>) -> Tensor {
        let mut x = Tensor::new([0.0]);
        let mut output = vec![];

        let mut _hidden = context_vector;
        for _ in 0..self.length {
            let embedded = self.embedding_decoder.forward(&x);
            let embedded = self.layer_norm_decoder.forward(&embedded);
            let out = self.gru_decoder.forward(&embedded, _hidden);

            let linear = self.linear_decoder.forward(&out);
            x = linear.argmax(-1);

            output.push(linear);

            _hidden = Some(out);
        }

        output.concat_tensor(0)
    }
}
