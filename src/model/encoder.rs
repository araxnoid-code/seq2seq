use burn::{
    config::Config,
    module::Module,
    nn::{Embedding, EmbeddingConfig, Lstm, LstmConfig, LstmState},
    prelude::Backend,
    tensor::{Int, Tensor},
};

#[derive(Debug, Module)]
pub struct Encoder<B: Backend> {
    embedding: Embedding<B>,
    lstm: Lstm<B>,
}

impl<B: Backend> Encoder<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Option<LstmState<B, 2>> {
        let embedded = self.embedding.forward(input);
        let mut state_save: Option<LstmState<B, 2>> = None;
        for i in 0..20 {
            let embedded_slice = embedded.clone().slice([0..1, i..i + 1]);
            if let Some(state) = state_save {
                let (_, state) = self.lstm.forward(embedded_slice, Some(state));
                state_save = Some(state);
            } else {
                let (_, state) = self.lstm.forward(embedded_slice, None);
                state_save = Some(state);
            }
        }

        state_save
    }
}

#[derive(Debug, Config)]
pub struct EncoderConfig {
    input: usize,
    hidden: usize,
}

impl EncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Encoder<B> {
        Encoder {
            embedding: EmbeddingConfig::new(self.input, self.hidden).init(device),
            lstm: LstmConfig::new(self.hidden, self.hidden, true).init(device),
        }
    }
}
