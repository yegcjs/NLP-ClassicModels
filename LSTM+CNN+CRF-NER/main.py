import argparse
import DataManager
import Loss
import Model
import torch
import numpy as np
import os
import random
import pdb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="conll2003")
    parser.add_argument("--vs", default=30000, type=int)
    parser.add_argument("--bs", default=128, type=int)
    parser.add_argument("--embedw", default="/home/yejs/pretrained/glove/glove.twitter.27B.100d.txt")
    parser.add_argument("--embedc_dim", default=30, type=int)
    parser.add_argument("--kernel", default=3, type=int)
    parser.add_argument("--hdim", default=200, type=int)
    parser.add_argument("--droprate", default=0.5, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--earlystop", default=5, type=int)
    parser.add_argument("--archive", default="debug.pt")
    parser.add_argument("--seed", default=803, type=int)
    parser.add_argument("--pretrained", default="none")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    os.environ['PATHOGENS'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    data = DataManager.DataManager(args.dataset, args.vs, args.bs, device=args.device)
    word_embeddings = data.load_pretrained_embeddings(args.embedw)
    for part, dataset in data.datasets.items():
        print(f"{part}: {len(dataset)}")

    model = Model.NERModel(
        word_embeddings, data.tokenizer.word2idx['<PAD>'],
        data.tokenizer.char_vocab_size, args.embedc_dim, data.tokenizer.char2idx['<PAD>'],
        data.tokenizer.wordlen, args.kernel,
        args.hdim, data.tokenizer.num_tags, data.tokenizer.tag2idx['<S>'], data.tokenizer.tag2idx['</S>'],
        dropout=args.droprate, device=args.device
    )
    # print(model.state_dict())
    # print("=======")
    # torch.load(args.pretrained, map_location=args.device)
    # exit()

    loss_function = Loss.CRFLoss(args.device, data.tokenizer.tag2idx['<S>'], data.tokenizer.tag2idx['</S>'])
    if args.pretrained != "none":
        model.load_state_dict(torch.load(args.pretrained, map_location=args.device))
    model.fit(data.load('train'), data.load('valid'), loss_function,
              args.epochs, args.earlystop, args.archive)
    # else:
    with torch.no_grad():
        print("Training Set")
        model.test(data.load('train'), data.tokenizer.idx2tag)
        print("Dev Set")
        model.test(data.load("valid"), data.tokenizer.idx2tag)
        print("Test Set")
        model.test(data.load('test'), data.tokenizer.idx2tag, idx2word=data.tokenizer.idx2word)


if __name__ == '__main__':
    _args = parse_args()
    torch.set_default_dtype(torch.float64)
    set_seed(_args.seed)
    main(_args)
