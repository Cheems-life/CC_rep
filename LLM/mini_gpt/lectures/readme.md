## bigram.py

二元语言模型，输入是一大段莎士比亚文本，程序首先把字符转换成数字，训练模型学习：

看到 'F' 下一个字符可能是什么？
...
最终模型可以自己不断预测


1. 超参数：
一次训练 32 条数据，每条序列长度是 8，也就是 32 条样本，每条 8 个 token
x.shape = (32, 8)


2. 读入数据，构造字符词表：
text 是一个巨大的 python 字符词，chars = sorted(list(set(text)))

假设 text = "hello"   那么 set(text) 得到 {'h', 'e', 'l', 'o'}  然后排序，chars = ['e', 'h', 'l', 'o']
tiny Shakespeare 大约有 65 种字符，因此 vocab_size = 65


3. 之后 get_batch 部分：
首先随机生成 32 个起点，x 是每个起点拿起 8 个字符，y 是整体向右移动一个字符
假设原文是：First Citizen   x 和 y 分别得到：
x:
F i r s t   C i

y:
i r s t   C i t

所以就是输入 x = 'F'  预测目标 y = 'i'   ...


4. 模型：BigramLanguageModel
每一个字符，都对应一个长度为 65 的向量，这个向量表示，如果当前字符是它，那么下一个字符分别是 65 个字符的倾向多大

B	batch size，同时处理多少条序列                    比如 32
T	time / sequence length，当前上下文的 token 数    比如 8
C	channel / vocab size，每个 token 的 logits 数	比如 65


(B,T,C)  ->  (32,8,65)   32 条序列 * 8 个序列 * 65 个候选字符的分数

例如某个位置输出：[-1.2, 0.3, 2.1, -0.5, ...]  这还不是概率，叫 logits，要经过 softmax 变为概率

5. reshape 是适应 cross_entropy，最后：
logits
(256,65)

targets
(256)

意思是：一共有 256 道“下一个字符是什么”的选择题，每一道有 65 个候选答案

6. 训练模型

补充：forward 不是手动调的，而是通过 model() 触发

为什么叫 Bigram？  因为它只学习两个 token 之间的关系：当前 token -> 下一个 token

7. generate
logits, loss = self(idx)  假设 idx = [\n]   得到 \n 后面 65 个字符分别有多大可能性

logits[:, -1, :]    保留所有 batch、只取序列最后一个时间步、保留完整词表维度，得到"下一个 token 的预测分布"

训练 3000 次，生成 500 个新 token