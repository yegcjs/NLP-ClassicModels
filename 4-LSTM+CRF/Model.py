import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pdb


class CharCNN(nn.Module):
    def __init__(self, num_chars, max_char_len, embed_dim, pad_idx, kernel_size, dropout=0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.char_len = max_char_len
        self.embedding = nn.Embedding(num_chars, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(p=dropout)
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size - 1
        )
        self.output_len = max_char_len + kernel_size - 1

    def forward(self, X):
        """
        :param X: chars, bsz x seq_len x max_char_len
        :return: word_embeddings: bsz x seq_len x embed_dim
        """
        bsz, seq_len = X.shape[0], X.shape[1]

        embeddings = self.dropout(self.embedding(X))  # bsz x seq_len x char_len x embed_dim
        perm_embeddings = embeddings.permute(0, 1, 3, 2).reshape(-1, self.embed_dim, self.char_len)
        conv_embeddings = self.conv(perm_embeddings)  # (bsz x seq_len), embed_dim, output_len
        pooled_embeddings = conv_embeddings.max(dim=-1)[0]  # （bsz x seq_len), embed_dim
        return pooled_embeddings.reshape(bsz, seq_len, self.embed_dim)


class CRF(nn.Module):
    def __init__(self, hidden_dim, num_tags, start_id, end_id):
        super().__init__()
        self.linear = nn.Linear(2 * hidden_dim, num_tags * num_tags)
        # self.transfer = torch.rand(num_tags, num_tags, requires_grad=True)
        # self.transfer[:, start_id] = -1e12
        # self.transfer[end_id, :] = -1e12
        # self.transfer[end_id, end_id] = 0
        # self.transfer = nn.Parameter(self.transfer)
        # self.register_parameter('transfer', self.transfer)

        self.num_tags = num_tags
        self.start_id = start_id
        self.end_id = end_id

    def forward(self, hidden_states, lengths=None):
        """
        :param lengths: length of sequences
        :param hidden_states: bsz, seq_len, hidden_dim
        :return: score_matrix, Z, [ S(x,y) if sequences given]
        """

        # transition = self.linear(hidden_states).unsqueeze(2).repeat(1, 1, self.num_tags, 1) + self.transfer
        bsz, seqlen = hidden_states.shape[0], hidden_states.shape[1]
        transition = self.linear(hidden_states).reshape(bsz, seqlen, self.num_tags, self.num_tags)
        # pdb.set_trace()
        # .reshape(bsz, seq_len, self.num_tags, self.num_tags)
        # [seq_len:] -> no more transition, except for </S>-></S>
        if lengths is not None:
            for i, length in enumerate(lengths):
                transition[i, length:, :, :] = -1e12
        transition[:, :, self.end_id, self.end_id] = 0  # end -> end: do_nothing
        transition[:, :, :, self.start_id] = -1e12  # any -> start: impossible
        transition[:, 1:, self.start_id, :] = -1e12
        return transition


class NERModel(nn.Module):
    def __init__(self,
                 pretrain_word_embedding, word_pad_id,  # word_embed
                 num_chars, char_embed_dim, char_pad_id, max_char_len, cnn_kernel_size,  # char cnn
                 hidden_dim, num_tags, start_tag_id, end_tag_id,  # lstm, crf
                 dropout=0.5, device="cpu"  # other optional params
                 ):
        super(NERModel, self).__init__()
        self.word_embed = nn.Embedding.from_pretrained(pretrain_word_embedding, padding_idx=word_pad_id, freeze=False)
        self.dropout = nn.Dropout(p=dropout)
        self.charcnn = CharCNN(num_chars, max_char_len, char_embed_dim, char_pad_id, cnn_kernel_size, dropout=dropout)

        self.vocab_size, self.word_embed_dim = pretrain_word_embedding.shape
        self.bilstm = nn.LSTM(self.word_embed_dim + char_embed_dim, hidden_dim,
                              bidirectional=True, dropout=dropout, batch_first=True)
        self.crf = CRF(hidden_dim, num_tags, start_tag_id, end_tag_id)

        self.num_tags, self.start_tag_id, self.end_tag_id = num_tags, start_tag_id, end_tag_id
        self.device = device
        self.to(device)
        self.optimizer = torch.optim.AdamW([
            {'params': [p for p in self.word_embed.parameters()], 'lr': 0.001},
            {'params': [p for p in self.charcnn.parameters()], 'lr': 0.001},
            {'params': [p for p in self.bilstm.parameters()], 'lr': 0.001},
            {'params': [p for p in self.crf.parameters()], 'lr': 0.001}
        ])

    def forward(self, sequences, lengths, chars):
        """
        :param sequences: bsz x seq_len
        :param lengths: list([length])
        :param chars: bsz x seq_len x max_char_len
        :return: transition matrix
        """
        word_embedding = self.dropout(self.word_embed(sequences))  # bsz, seq_len, word_embed_dim
        char_embedding = self.charcnn(chars)  # bsz, seq_len, char_embed_dim
        embedding = pack_padded_sequence(torch.cat([word_embedding, char_embedding], dim=-1), lengths,
                                         batch_first=True, enforce_sorted=False)
        output, _ = self.bilstm(embedding)
        hidden, _ = pad_packed_sequence(output, batch_first=True)  # bsz x seq_len, 2*hidden_dim
        hidden = self.dropout(hidden)
        return self.crf(hidden, lengths=lengths)  # transition matrix, target_score

    def predict(self, sequences, lengths, chars):
        self.eval()
        transition = self.forward(sequences, lengths, chars)  # bsz, seqlen, num_tags, num_tags
        bsz, seqlen = transition.shape[0], transition.shape[1]
        scores, prev_tag_ids = torch.zeros(bsz, seqlen, self.num_tags, device=transition.device), \
                               torch.zeros(bsz, seqlen, self.num_tags, dtype=torch.int64, device=transition.device)

        scores[:, 0, :] = transition[:, 0, self.start_tag_id, :]
        prev_tag_ids[:, 0, :] = self.start_tag_id

        idxes = torch.tensor(range(bsz), device=transition.device)
        for t in range(1, seqlen):
            # bsz, num_tags, num_tags ->
            score_tmp = scores[:, t - 1, :].unsqueeze(2).repeat(1, 1, self.num_tags) + transition[:, t, :, :]
            scores[:, t, :], prev_tag_ids[:, t, :] = score_tmp.max(dim=1)  # bsz, num_tags, num_tags -> bsz, num_tags

        prediction = torch.zeros(bsz, seqlen, dtype=torch.int64, device=transition.device)
        prediction[:, seqlen - 1] = self.end_tag_id
        for t in range(seqlen - 2, -1, -1):
            prediction[:, t] = prev_tag_ids[idxes, t + 1, prediction[:, t + 1]]
        return prediction

    def fit(self, train_data, eval_data, loss_function, num_epochs, early_stop, archivedir):
        early_stop_buf = 0
        min_eval_loss = 999999

        for epoch in range(num_epochs):
            train_loss = []
            eval_loss = []
            self.train()
            for sequences, lengths, chars, tags in train_data:
                transition_matrix = self.forward(sequences, lengths, chars)
                loss = loss_function(transition_matrix, tags)
                train_loss.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(self.parameters(), 5)
                # pdb.set_trace()
                self.optimizer.step()

            self.eval()
            with torch.no_grad():
                for sequences, lengths, chars, tags in eval_data:
                    transition_matrix = self.forward(sequences, lengths, chars)
                    loss = loss_function(transition_matrix, tags)
                    eval_loss.append(loss.item())

            train_loss = sum(train_loss) / len(train_loss)
            eval_loss = sum(eval_loss) / len(eval_loss)
            print(f"Epoch {epoch}\n Train: {train_loss}, Eval: {eval_loss}")

            if eval_loss < min_eval_loss:
                min_eval_loss = eval_loss
                torch.save(self.state_dict(), archivedir)
                early_stop_buf = 0
            else:
                early_stop_buf += 1
                if early_stop_buf >= early_stop:
                    print("Early Stop!")
                    break
        self.load_state_dict(torch.load(archivedir, map_location=self.device))

    def test(self, data, idx2tag, idx2word=None):
        """
        :param data: test data
        :return: accuracy, precision, recall, f1
        """
        cnt_tot, cnt_correct = 0, 0
        true_pos, pred_pos, ground_pos = 0, 0, 0

        for sequences, lengths, chars, _tags in data:
            bsz = sequences.shape[0]
            _predictions = self.predict(sequences, lengths, chars)

            predictions = [[idx2tag[predtag] for predtag in prediction] for prediction in _predictions]
            tags = [[idx2tag[tag] for tag in seqtag] for seqtag in _tags]

            if idx2word is not None:
                with open("test_output.txt", 'w', encoding='utf-8') as f:
                    for seq, seqtags, preds in zip(sequences, tags, predictions):
                        for token, tag, pred in zip(seq, seqtags, preds):
                            if idx2word[token] == '</S>':
                                f.write('\n')
                                break
                            f.write(f"{idx2word[token]} {tag} {pred}\n")

            STATE_OUTSIDE, STATE_IN_ENTIRY = 0, 1
            for i in range(bsz):
                state = STATE_OUTSIDE
                i_tag, e_tag = ' ', ' '
                for j, (prediction, tag) in enumerate(zip(predictions[i], tags[i])):
                    # pdb.set_trace()
                    if tag == '</S>': break
                    # acc
                    cnt_tot += 1
                    if prediction == tag: cnt_correct += 1

                    if state == STATE_IN_ENTIRY:
                        if prediction == i_tag or prediction == e_tag:
                            if prediction == tag:
                                state = STATE_IN_ENTIRY if tag == i_tag else STATE_OUTSIDE
                                continue
                            else:
                                true_pos -= 1
                                state = STATE_OUTSIDE
                        else:  # break the rule
                            true_pos -= 1
                            state = STATE_OUTSIDE

                    if 'B' or 'S' in tag:
                        ground_pos += 1
                    if 'B' or 'S' in prediction:
                        pred_pos += 1
                        assert state == STATE_OUTSIDE
                        if tag == prediction:
                            true_pos += 1
                            if 'B' in prediction:
                                state = STATE_IN_ENTIRY
                                i_tag = 'I' + prediction[1:]
                                e_tag = 'E' + prediction[1:]

                    # pdb.set_trace()

        accuracy = cnt_correct / cnt_tot
        precision = true_pos / pred_pos
        recall = true_pos / ground_pos
        f1 = 2 * precision * recall / (precision + recall)

        print(f"ACC: {accuracy}, Prec: {precision}, Rec: {recall}, F1: {f1}")
        return accuracy, precision, recall, f1
