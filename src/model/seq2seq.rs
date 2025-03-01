use burn::{
    config::Config,
    module::Module,
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig, Lstm, LstmConfig,
        LstmState,
    },
    prelude::Backend,
    tensor::{activation::softmax, Int, Tensor},
};

#[derive(Debug, Module)]
pub struct Seq2Seq<B: Backend> {
    // encoder
    encoder_embedding: Embedding<B>,
    encoder_dropout: Dropout,
    encoder_lstm: Lstm<B>,

    //decoder encoder
    decoder_embedding: Embedding<B>,
    decoder_dropout: Dropout,
    decoder_lstm: Lstm<B>,
    decoder_linear: Linear<B>,
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

    pub fn decoder_forward(&self, state: LstmState<B, 2>) -> Tensor<B, 2> {
        let input: Tensor<B, 2, Int> = Tensor::from([[0]]);
        let mut state = state;

        let mut outputs = vec![];
        for i in 0..12 {
            let embedded = self
                .decoder_dropout
                .forward(self.decoder_embedding.forward(input.clone()));

            let (lstm_output, lstm_state) = self.decoder_lstm.forward(embedded, Some(state));
            state = lstm_state;

            let output_linear: Tensor<B, 2> = self.decoder_linear.forward(lstm_output).squeeze(0);

            outputs.push(output_linear);
        }
        let outputs = Tensor::cat(outputs, 0);

        outputs
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
            //encoder
            encoder_embedding: EmbeddingConfig::new(self.input, self.hidden).init(device),
            encoder_dropout: DropoutConfig::new(self.dropout).init(),
            encoder_lstm: LstmConfig::new(self.hidden, self.hidden, true).init(device),

            // decoder
            decoder_embedding: EmbeddingConfig::new(self.input, self.hidden).init(device),
            decoder_dropout: DropoutConfig::new(self.dropout).init(),
            decoder_lstm: LstmConfig::new(self.hidden, self.hidden, true).init(device),
            decoder_linear: LinearConfig::new(self.hidden, self.output).init(device),
        }
    }
}
