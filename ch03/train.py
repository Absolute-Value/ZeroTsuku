import sys
sys.path.append('..')
from common.trainer import Trainer
from common.optimizer import Adam
from simple_cbow import SimpleCBOW
from common.util import preprocess, create_contexts_target, convert_one_hot
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--window_size', type=int, default=1)
parser.add_argument('--hidden_size', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=3)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument('--seed', type=int, default=111, help='manual seed')
args = parser.parse_args()

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

vocab_size = len(word_to_id)
contexts, target = create_contexts_target(corpus, args.window_size)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

model = SimpleCBOW(vocab_size, args.hidden_size)
optimizer = Adam(lr=args.lr)
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, args.epochs, args.batch_size)
trainer.plot()

word_vecs = model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])