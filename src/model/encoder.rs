use burn::{
    config::Config,
    module::Module,
    nn::{Embedding, EmbeddingConfig, Lstm, LstmConfig},
    prelude::Backend,
    tensor::{Int, Tensor},
};

#[derive(Debug, Module)]
pub struct Encoder<B: Backend> {
    embedding: Embedding<B>,
    lstm: Lstm<B>,
}

impl<B: Backend> Encoder<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> (Tensor<B, 3>, burn::nn::LstmState<B, 2>) {
        let embedded = self.embedding.forward(input);
        let (context_vector, state) = self.lstm.forward(embedded, None);
        (context_vector, state)
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
