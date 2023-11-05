import pandas as pd
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import random

from src import ROOT_DIR

teacher_forcing_ratio = 0.5  # global variable
# Step: 5.
emb_dim = 256
hid_dim = 512
n_layers = 2
dropout = 0.5
num_tokens = 0  # just a declaration


def tokenize(sentence, vocab, max_length):
    return [vocab.get(token, vocab['<unk>']) for token in sentence.lower().split()[:max_length]]


class TextDataset(Dataset):
    def __init__(self, input_texts, target_texts, vocab, max_length=512):
        self.input_texts = [torch.tensor(tokenize(text, vocab, max_length)) for text in input_texts]
        self.target_texts = [torch.tensor(tokenize(text, vocab, max_length)) for text in target_texts]
        self.pad_idx = vocab['<pad>']

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        return self.input_texts[idx], self.target_texts[idx]

    def collate_fn(self, batch):
        input_texts, target_texts = zip(*batch)
        # Padding sequences to the max length in each batch
        input_texts = pad_sequence(input_texts, batch_first=True, padding_value=self.pad_idx)
        target_texts = pad_sequence(target_texts, batch_first=True, padding_value=self.pad_idx)
        return input_texts, target_texts


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        # Embedding
        embedded = self.dropout(self.embedding(src))

        # Pack sequence
        packed_embedded = pack_padded_sequence(embedded, src_len.to('cpu'), batch_first=True, enforce_sorted=False)

        # Pass packed sequence through rnn
        packed_outputs, (hidden, cell) = self.rnn(packed_embedded)

        # Unpack sequence
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)

        # outputs is now a padded sequence
        return outputs, hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        embedded = self.dropout(self.embedding(input))

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        prediction = self.fc_out(output.squeeze(1))

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, src_len):
        batch_size = src.shape[0]
        trg_len = trg.shape[1] if trg is not None else None
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # Encode source sequence
        encoder_outputs, hidden, cell = self.encoder(src, src_len)

        # First input to the decoder is the <sos> token, assume it's always the first in the vocab
        input = trg[:, 0]

        for t in range(1, trg_len):
            # Decode one step at a time
            # The decoder takes in the previous target token and the hidden states
            # We unsqueeze(1) to add the sequence length dimension (which is 1)
            output, hidden, cell = self.decoder(input.unsqueeze(1), hidden, cell)

            # Save the output
            outputs[:, t, :] = output.squeeze(1)

            # Get the most probable next token
            top1 = output.squeeze(1).argmax(1)

            # Decide whether to use teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else top1

        return outputs


def training_loop(model, train_dataloader, device, vocab, optimizer, criterion, val_dataloader, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        # Training loop
        for src, trg in train_dataloader:
            src, trg = src.to(device), trg.to(device)
            assert trg.ndim == 2, "Target tensor trg should have 2 dimensions [batch_size, sequence_length]"
            src_len = torch.sum(src != vocab['<pad>'], dim=1)
            # src: [batch_size, src_len], trg: [batch_size, trg_len]
            src, trg = src.to(device), trg.to(device)
            # Calculate the length of each sentence in the src batch
            src_len = torch.sum(src != vocab['<pad>'], dim=1)  # [batch_size]
            optimizer.zero_grad()
            # Forward pass
            output = model.forward(src, trg,
                                   src_len)  # trg is not shifted here, assuming trg[:, 0] is <sos> in Seq2Seq model

            # trg is shifted inside the model, so we don't consider the first token (<sos>) in the loss
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            # Calculate loss
            loss = criterion(output, trg)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        average_train_loss = total_train_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1} Train Loss: {average_train_loss:.4f}')

        # Evaluation loop
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for src, trg in val_dataloader:
                src, trg = src.to(device), trg.to(device)
                src_len = torch.sum(src != vocab['<pad>'], dim=1)  # [batch_size]

                # Forward pass
                output = model.forward(src, trg, src_len)

                # Calculate loss
                output = output[:, 1:].reshape(-1, output_dim)
                trg = trg[:, 1:].reshape(-1)
                loss = criterion(output, trg)
                total_val_loss += loss.item()

        average_val_loss = total_val_loss / len(val_dataloader)
        print(f'Epoch {epoch + 1} Validation Loss: {average_val_loss:.4f}')


def main(path: str = f'{ROOT_DIR}/data/filtered_and_prepocessed.csv'):
    data = pd.read_csv(path)

    input_texts = list(data['reference'])
    target_texts = list(data['translation'])

    # Step: 1. Focusing on str objects only
    input_texts = list(filter(lambda x: isinstance(x, str), input_texts))[:1000]
    target_texts = list(filter(lambda x: isinstance(x, str), target_texts))[:1000]

    # Step: 2. Gathering own vocab
    word_counter = Counter()

    for sentence in input_texts + target_texts:
        word_counter.update(sentence.lower().split())

    vocab = {word: index + 4 for index, word in enumerate(word_counter)}  # +4 for special tokens
    vocab['<pad>'] = 0
    vocab['<unk>'] = 1
    vocab['<eos>'] = 2
    vocab['<sos>'] = 3

    num_tokens = len(vocab)

    # Step: 3. Split the dataset
    input_train, input_val, target_train, target_val = train_test_split(input_texts, target_texts, test_size=0.1)

    # Step: 4. Create datasets and dataloaders
    train_dataset = TextDataset(input_train, target_train, vocab, max_length=500)
    val_dataset = TextDataset(input_val, target_val, vocab, max_length=500)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=train_dataset.collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=32, collate_fn=val_dataset.collate_fn)

    # Step: 5. Initializing hyper-params
    # above. outside this func

    # Step: 6. Initialize encoder and decoder
    encoder = Encoder(num_tokens, emb_dim, hid_dim, n_layers, dropout)
    decoder = Decoder(num_tokens, emb_dim, hid_dim, n_layers, dropout)

    # Step: 7. Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Step: 8. Initialize the seq2seq model
    model = Seq2Seq(encoder, decoder, device).to(device)

    # Step: 9. Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Step: 10. Define the loss function, ignoring the padded elements in the output sequence
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])

    # Step: 11. Train loop
    training_loop(model, train_dataloader, device, vocab, optimizer, criterion, val_dataloader)

    # Step: 12. Saving
    torch.save(model.state_dict(), f'{ROOT_DIR}/models/seq2seq_model.pth')
    torch.save(model, f'{ROOT_DIR}/models/seq2seq_model_complete.pth')
    return vocab


if __name__ == '__main__':
    vocab = main()

    # Step: 13. Passing vocab for predicting
    vocab_inv = {index: token for token, index in vocab.items()}

    df_vocab_inv = pd.DataFrame(list(vocab_inv.items()), columns=['index', 'token'])
    df_vocab_inv.to_csv(f'{ROOT_DIR}/data/vocab_inv.csv')
