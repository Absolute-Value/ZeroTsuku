import sys
sys.path.append('..')
from dataset import ptb

corpus, word_to_id, id_to_word = ptb.load_data('train')

print('corpus size:', len(corpus))
print('corpus[:30]:', corpus[:30])
print()
for i in range(3):
    print(f'id_to_word[{i}]:', id_to_word[i])
for word in ['car', 'happy', 'lexus']:
    print(f'word_to_id[{word}]', word_to_id[word])