import torch
from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import pdb


class Tokenizer:
    def __init__(self, sequences, sequence_tags, vocab_size, device="cpu"):
        """
        :param sequences: list([list([tokens])])
        :param sequence_tags: list([list([tags])])
        initialize: word2idx, idx2word; tag2idx, idx2tag; char2idx, idx2char
        """
        vocab_cnt = Counter([word for seq in sequences for word in seq]).most_common(vocab_size)
        self.idx2word = ['<PAD>'] + [word for word, cnt in vocab_cnt] + ['</S>', '<UNK>']
        self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}
        self.idx2char = ['<PAD>'] + list(Counter([c for word in self.idx2word for c in list(word)]).keys()) + ['<UNK>']
        self.char2idx = {c: idx for idx, c in enumerate(self.idx2char)}

        # </S> as pad
        self.idx2tag = ['</S>'] + list(Counter([tag for tags in sequence_tags for tag in tags]).keys()) + ['<S>']
        self.tag2idx = {tag: idx for idx, tag in enumerate(self.idx2tag)}

        self.device = device
        self.wordlen = max(len(word) for word in self.idx2word)

    @property
    def num_tags(self):
        return len(self.idx2tag)

    @property
    def char_vocab_size(self):
        return len(self.idx2char)

    @property
    def word_vocab_size(self):
        return len(self.idx2word)

    def tokenize(self, sequences, sequence_tags, device=None):
        """
        :param sequences: list([list([tokens])])
        :param sequence_tags: list([list([tags])])
        :param device: device, default: tokenizer's device
        :return: convert strings/chars into index representation
        """
        if device is None:
            device = self.device

        sequences = [seq + ['</S>'] for seq in sequences if len(seq)]  # xxx
        sequence_tags = [tags + ['</S>'] for tags in sequence_tags if len(tags)]

        sequences_ids = [torch.tensor([
            self.word2idx[word] if word in self.word2idx else self.word2idx['<UNK>'] for word in sequence
        ]) for sequence in sequences]
        sequences_ids = pad_sequence(sequences_ids, batch_first=True).to(device)  # bsz, max_seq_len
        sequences_length = [len(sequence) for sequence in sequences]

        num_sequences = len(sequences)
        # max_char_len = max([len(word) for sequence in sequences for word in sequence])
        char_ids = torch.zeros(num_sequences, max(sequences_length), self.wordlen, dtype=torch.int64)
        for i, sequence in enumerate(sequences):
            for j, word in enumerate(sequence):
                for k, c in enumerate(list(word)):
                    char_ids[i, j, k] = self.char2idx[c] if c in self.char2idx else self.char2idx['<UNK>']
        char_ids = char_ids.to(device)

        tags_ids = pad_sequence([
            torch.tensor([self.tag2idx[tag] for tag in tags]) for tags in sequence_tags
        ], padding_value=self.tag2idx['</S>'], batch_first=True).to(device)
        return sequences_ids, sequences_length, char_ids, tags_ids


class NERDataset(Dataset):
    def __init__(self, sequences_ids, sequences_length, char_ids, tags_ids):
        """
        :param sequences_ids: n_sequences x max_seq_len
        :param sequences_length: list([length])
        :param char_ids: n_sequences x max_seq_len x max_char_len
        :param tags_ids: list([tensor(length)])
        """
        super().__init__()
        self.seq = sequences_ids
        self.length = sequences_length
        self.char = char_ids
        self.tags = tags_ids

        self.size = self.seq.shape[0]

    def __getitem__(self, idx):
        return self.seq[idx], self.length[idx], self.char[idx], self.tags[idx]

    def __len__(self):
        return self.size

    def to(self, device):
        self.seq = self.seq.to(device)
        self.char = self.char.to(device)
        for seq_tags in self.tags:
            seq_tags = seq_tags.to(device)
        return self


class DataManager:
    def __init__(self, datadir, vocab_size=30000, batch_size=128, device="cpu"):
        parts = ['train', 'valid', 'test']
        train_sequences, train_seq_tags = self.__read_data(f"{datadir}/train.txt")
        dev_sequences, dev_seq_tags = self.__read_data(f"{datadir}/valid.txt")
        test_sequences, test_seq_tags = self.__read_data(f"{datadir}/test.txt")
        all_sequences, all_seq_tags = train_sequences + dev_sequences + test_sequences, \
                                      train_seq_tags + dev_seq_tags + test_seq_tags
        tokenizer = Tokenizer(all_sequences, all_seq_tags, vocab_size, device=device)

        self.datasets = {}
        for part in parts:
            sequences, seq_tags = self.__read_data(f"{datadir}/{part}.txt")
            seq_ids, seq_len, char_idx, seq_tags = tokenizer.tokenize(sequences, seq_tags)
            self.datasets[part] = NERDataset(seq_ids, seq_len, char_idx, seq_tags)
        self.bsz = batch_size
        self.tokenizer = tokenizer
        self.device = device

    def __read_data(self, path):
        """
        :param path: path of data file
        :return: list([list([word])]) sequences
        :return: list([list([tag])]) sequences_tags
        """
        sequences, sequences_tags = [], []
        with open(path, 'r', encoding='utf-8') as f:
            sequence, seq_tags = [], []
            for line in f:
                if line == '\n':
                    sequences.append(sequence)
                    sequences_tags.append(seq_tags)
                    sequence = []
                    seq_tags = []
                else:
                    word, _, _, tag = line.strip().split()
                    sequence.append(word)
                    seq_tags.append(tag)
        return sequences, sequences_tags

    def load(self, part, shuffle=True, batch_size=None):
        if batch_size is None:
            batch_size = self.bsz
        return DataLoader(dataset=self.datasets[part], shuffle=shuffle, batch_size=batch_size)

    def load_pretrained_embeddings(self, path, device=None):
        if device is None:
            device = self.device

        with open(path, 'r', encoding='utf-8') as f:
            lines = [line.strip().split() for line in f]
            embed_dim = len(lines[0]) - 1

        word2embed = {line[0]: list(map(float, line[1:])) for line in lines}

        embedding = torch.tensor([
            word2embed[word] if word in word2embed else torch.rand(embed_dim).tolist()
            for word in self.tokenizer.idx2word
        ]).to(device)
        return embedding
