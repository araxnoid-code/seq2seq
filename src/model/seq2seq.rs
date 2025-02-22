use burn::{
    config::Config,
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, Lstm, LstmConfig, LstmState},
    prelude::Backend,
    tensor::{
        activation::{relu, softmax},
        Int, Tensor,
    },
};

#[derive(Debug, Module)]
pub struct Seq2Seq<B: Backend> {
    // encoder
    encoder_embedding: Embedding<B>,
    encoder_lstm: Lstm<B>,
    //decoder
    decoder_embedding: Embedding<B>,
    decoder_lstm: Lstm<B>,
    decoder_linear: Linear<B>,
    decoder_linear_2: Linear<B>,
}

impl<B: Backend> Seq2Seq<B> {
    pub fn encoder_forward(
        &self,
        input: Tensor<B, 2, Int>,
    ) -> (Option<LstmState<B, 2>>, Vec<Tensor<B, 2>>) {
        let embedded = self.encoder_embedding.forward(input);
        let mut state_save: Option<LstmState<B, 2>> = None;
        let mut hiddens = vec![];
        for i in 0..20 {
            let embedded_slice = embedded.clone().slice([0..1, i..i + 1]);
            if let Some(state) = state_save {
                let (_, state) = self.encoder_lstm.forward(embedded_slice, Some(state));
                // save for attention
                hiddens.push(state.hidden.clone());
                //
                state_save = Some(state);
            } else {
                let (_, state) = self.encoder_lstm.forward(embedded_slice, None);
                // save for attention
                hiddens.push(state.hidden.clone());
                //
                state_save = Some(state);
            }
        }

        (state_save, hiddens)
    }

    pub fn decoder_forward(
        &self,
        hiddens: Vec<Tensor<B, 2>>,
        state: LstmState<B, 2>,
        teaching: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 2> {
        let mut outputs = Vec::new();
        if let Some(target) = teaching.clone() {
            let input_model = vec![Tensor::from([[0]]), target.clone()];
            let input_model = Tensor::cat(input_model, 1);
            let embedded = self.decoder_embedding.forward(input_model);
            let mut state_model = state;
            for i in 0..21 {
                let embedded_slice = embedded.clone().slice([0..1, i..i + 1]);
                let (output, state) = self.decoder_lstm.forward(embedded_slice, Some(state_model));
                // let output = self.decoder_linear.forward(output);
                // let output = relu(output);
                self.attention(output.clone(), hiddens.clone());
                let output = self.decoder_linear_2.forward(output);
                // let output = relu(output);
                let output = output.reshape([0, -1]);
                let output = softmax(output, 1);
                outputs.push(output);
                state_model = state;
            }
        } else {
            let mut input: Tensor<B, 2, Int> = Tensor::from([[0]]);
            let mut state_model = state;
            for _ in 0..21 {
                let embedded = self.decoder_embedding.forward(input.clone());
                let (output, state) = self.decoder_lstm.forward(embedded, Some(state_model));
                // let output = self.decoder_linear.forward(pred);
                // let output = relu(output);
                let output = self.decoder_linear_2.forward(output);
                // let output = relu(output);
                let output = output.reshape([0, -1]);
                let output = softmax(output, 1);
                outputs.push(output.clone());

                // update input
                let next_input = output.argmax(1);
                input = next_input;
                state_model = state
            }
        }

        Tensor::cat(outputs, 0)
    }

    pub fn attention(&self, tensor: Tensor<B, 3>, hiddens: Vec<Tensor<B, 2>>) {
        let tensor = tensor.reshape([0, -1]);
        let mut matmuls = vec![];
        for hidden in hiddens {
            let hidden = hidden.permute([1, 0]);
            let matmul = tensor.clone().matmul(hidden);
            matmuls.push(matmul);
        }
        // matmuls.
    }

    pub fn forward(
        &self,
        input: Tensor<B, 2, Int>,
        teaching: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 2> {
        let (context_vector, hiddens) = self.encoder_forward(input.clone());

        let state = context_vector.unwrap();
        let pred = self.decoder_forward(hiddens, state, teaching);
        pred
    }
}

#[derive(Debug, Config)]
pub struct Seq2SeqConfig {
    input: usize,
    hidden: usize,
    output: usize,
}

impl Seq2SeqConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Seq2Seq<B> {
        Seq2Seq {
            // encoder
            encoder_embedding: EmbeddingConfig::new(self.input, self.hidden).init(device),
            encoder_lstm: LstmConfig::new(self.hidden, self.hidden, true).init(device),
            // decode.init
            decoder_embedding: EmbeddingConfig::new(self.output, self.hidden).init(device),
            decoder_lstm: LstmConfig::new(self.hidden, self.hidden, true).init(device),
            decoder_linear: LinearConfig::new(self.hidden, self.hidden).init(device),
            decoder_linear_2: LinearConfig::new(self.hidden, self.output).init(device),
        }
    }
}
