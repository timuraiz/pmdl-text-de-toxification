import torch
import pandas as pd

from src import ROOT_DIR
from train_model import Seq2Seq, Encoder, Decoder


def post_process(words):
    # Split the sentence into words and initialize an empty list for processed words
    processed_words = []
    not_allowed_words = ['<pad>', '<unk>', '<eos>', '<sos>']
    for word in words:
        if (not processed_words or word not in processed_words) and word not in not_allowed_words:
            processed_words.append(word)
    return processed_words


def translate_sentence(model, sentence, vocab, device, vocab_inv):
    model.eval()
    tokens = [vocab['<sos>']] + [vocab.get(word, vocab['<unk>']) for word in sentence.lower().split()] + [
        vocab['<eos>']]
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    src_len = torch.LongTensor([len(tokens)]).to(device)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor, src_len)

    trg_tokens = [vocab['<sos>']]

    for i in range(200):
        trg_tensor = torch.LongTensor([trg_tokens[-1]]).unsqueeze(0).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)

            # Ensure output is 2D before calling argmax
            output = output.squeeze(1) if output.dim() == 3 else output
            pred_token = output.argmax(1).item()
            trg_tokens.append(pred_token)

            if pred_token == vocab['<eos>']:
                break

    translated_sentence = [vocab_inv[token] for token in trg_tokens if token in vocab_inv]

    return post_process(translated_sentence)


def main(vocab, vocab_inv, sentence_to_translate="The service at this place was the worst I've ever experienced."):
    # Step: 1. Taking all notes from training model steps
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(f'{ROOT_DIR}/models/seq2seq_model_complete.pth', map_location=device)
    model = model.to(device)
    model.eval()

    # Step: 2. inverting to tokens
    translated_sentence_tokens = translate_sentence(model, sentence_to_translate, vocab, device, vocab_inv)

    # Step: 3. Printing result
    translated_sentence = ' '.join(translated_sentence_tokens)
    print(f"Original sentence: {sentence_to_translate}")
    print(f"Detoxified sentence: {translated_sentence}")


if __name__ == '__main__':
    df_vocab_inv = pd.read_csv(f'{ROOT_DIR}/data/vocab_inv.csv')
    vocab_inv = df_vocab_inv.set_index('index')['token'].to_dict()
    vocab = {token: idx for idx, token in vocab_inv.items()}
    main(vocab, vocab_inv)
