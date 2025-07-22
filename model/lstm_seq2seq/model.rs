struct MySeq2SeqModel {
    _model: Module,
    length: usize,
    // encoder
    embedding_encoder: Embedding,
    layer_norm_encoder: LayerNorm,
    lstm_encoder: Lstm,

    // decoder
    embedding_decoder: Embedding,
    layer_norm_decoder: LayerNorm,
    lstm_decoder: Lstm,
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
            lstm_encoder: model.lstm_init(hidden),

            // decoder
            embedding_decoder: model.embedding_init(vocab_num, hidden),
            layer_norm_decoder: model.layer_norm_init(&[hidden]),
            lstm_decoder: model.lstm_init(hidden),
            linear_decoder: model.liniar_init(hidden, vocab_num),

            // model
            _model: model,
            length,
        }
    }

    pub fn encoder(&mut self, x: &Tensor) -> Option<rotta_rs::LSTMCellHidden> {
        let embedded = self.embedding_encoder.forward(&x.reshape(vec![self.length as i32]));
        let embedded = self.layer_norm_encoder.forward(&embedded);

        let mut cell_hidden = None;
        for i in 0..self.length {
            let x = embedded.index(vec![i as i32]).reshape(vec![1, -1]);
            let out = self.lstm_encoder.forward(&x, cell_hidden);
            cell_hidden = Some(out);
        }

        cell_hidden
    }

    pub fn decoder(&mut self, context_vector: Option<LSTMCellHidden>) -> Tensor {
        let mut x = Tensor::new([0.0]);
        let mut output = vec![];

        let mut cell_hidden = context_vector;
        for _ in 0..self.length {
            let embedded = self.embedding_decoder.forward(&x);
            let embedded = self.layer_norm_decoder.forward(&embedded);
            let out = self.lstm_decoder.forward(&embedded, cell_hidden);

            let linear = self.linear_decoder.forward(&out.hidden);
            x = linear.argmax(-1);

            output.push(linear);

            cell_hidden = Some(out);
        }

        output.concat_tensor(0)
    }
}
