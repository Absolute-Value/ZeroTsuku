import sys
sys.path.append('..')
import numpy as np
from common.optimizer import SGD
from dataset import spiral
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet
import argparse

# ハイパーパラメータの設定
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--batch_size", type=int, default=30)
parser.add_argument("--hidden_size", type=int, default=10)
parser.add_argument("--lr", type=float, default=1.0)
parser.add_argument('--seed', type=int, default=111, help='manual seed')
args = parser.parse_args()

batch_size = args.batch_size

# データの読み込み、モデルとオプティマイザの生成
x, t = spiral.load_data()
model = TwoLayerNet(input_size=2, hidden_size=args.hidden_size, output_size=3)
optimizer = SGD(lr=args.lr)

# 学習で使用する変数
data_size = len(x)
max_iters = data_size // batch_size
total_loss = 0
loss_count = 0
loss_list = []

for epoch in range(1, args.epochs+1):
    # データのシャッフル
    idx = np.random.permutation(data_size)
    x = x[idx]
    t = t[idx]

    for iters in range(max_iters):
        batch_x = x[iters*batch_size:(iters+1)*batch_size]
        batch_t = t[iters*batch_size:(iters+1)*batch_size]

        # 勾配を求め、パラメータを更新
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)

        total_loss += loss
        loss_count += 1

        if (iters+1) % 10 == 0:
            avg_loss = total_loss / loss_count
            print(f"Epoch: {epoch}, Iter : {iters+1}, Loss : {avg_loss}")
            loss_list.append(avg_loss)
            total_loss, loss_count = 0, 0

plt.plot(loss_list, label='train_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()