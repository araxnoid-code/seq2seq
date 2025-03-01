use burn::{
    config::Config,
    nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, Lstm, LstmConfig, LstmState},
    prelude::Backend,
    tensor::{Int, Tensor},
};

pub struct Seq2Seq<B: Backend> {
    // encoder
    encoder_embedding: Embedding<B>,
    encoder_dropout: Dropout,
    encoder_lstm: Lstm<B>,
}

impl<B: Backend> Seq2Seq<B> {
    pub fn encoder_forward(&self, input: Tensor<B, 2, Int>) -> LstmState<B, 2> {
        let shape = input.dims();
        let mut state: Option<LstmState<B, 2>> = None;
        for i in 0..shape[shape.len() - 1] {
            let input = input.clone().slice([0..1, i..i + 1]);
            let embedded = self
                .encoder_dropout
                .forward(self.encoder_embedding.forward(input));

            let (_, lstm_state) = self.encoder_lstm.forward(embedded, state);
            state = Some(lstm_state);
        }
        state.unwrap()
    }
}

#[derive(Debug, Config)]
pub struct Seq2SeqConfig {
    input: usize,
    hidden: usize,
    output: usize,
    dropout: f64,
}

impl Seq2SeqConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Seq2Seq<B> {
        Seq2Seq {
            encoder_embedding: EmbeddingConfig::new(self.input, self.hidden).init(device),
            encoder_dropout: DropoutConfig::new(self.dropout).init(),
            encoder_lstm: LstmConfig::new(self.hidden, self.hidden, true).init(device),
        }
    }
}
