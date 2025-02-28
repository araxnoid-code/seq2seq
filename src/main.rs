mod tokenizing;
use tokenizing::*;

fn main() {
    let mut token = Tokenizing::default();
    token.get_data("data.txt");

    println!("{:?}", token.word2index);
}
