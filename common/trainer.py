import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from common.util import clip_grads
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []

    def fit(self, x, t, epochs=10, batch_size=32, max_grad=None, eval_interval=20):
        # 学習で使用する変数
        data_size = len(x)
        max_iters = data_size // batch_size
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        loop = tqdm(range(1, epochs+1))
        for epoch in loop:
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
                params, grads = remove_duplicate(model.params, model.grads)  # 共有された重みを1つに集約
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)

                total_loss += loss
                loss_count += 1

                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    avg_loss = total_loss / loss_count
                    #print(f"Epoch: {epoch}, Loss : {avg_loss}")
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0

    def plot(self):
        plt.plot(self.loss_list, label='train_loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig("loss.png")

def remove_duplicate(params, grads):
    '''
    パラメータ配列中の重複する重みをひとつに集約し、
    その重みに対応する勾配を加算する
    '''
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 重みを共有する場合
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 勾配の加算
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 転置行列として重みを共有する場合（weight tying）
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break

    return params, grads