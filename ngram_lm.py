import torch
import torch.nn as nn
import torch.nn.functional as F

class NGramLanguageModel(nn.Module):
    def __init__(self, vocab_size, context_size, embedding_dim, hidden_dim):
        super(NGramLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(context_size * embedding_dim, 128)
        self.fc2 = nn.Linear(128, vocab_size)


    def forward(self, inputs):
        embeds = self.embedding(inputs).view((1, -1))
        out = F.relu(self.fc1(embeds))
        out = self.fc2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

# Create a dataset of text
with open(r"C:\Users\Soumyadip Nandi\Downloads\policy\language\text.txt", 'r', encoding='utf-8') as f:
    text = f.read().split()
context_size = 2
train_data = []
n = 3  # Use trigrams
train_data = []
for i in range(n, len(text)):
    context = text[i - n:i - 1]
    target = text[i]
    train_data.append((context, target))

#print(train_data)

chars = sorted(list(set(text)))

vocab_size = len(chars)


# create a mapping from characters to integers
wti = { ch:i for i,ch in enumerate(chars) }
itw = { i:ch for i,ch in enumerate(chars) }

# print(wti)
# print(itw)

model = NGramLanguageModel(len(wti), context_size, 100, 128)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
epochs = 100

# Train the model
for epoch in range(epochs):
    total_loss = 0
    for context, target in train_data:
        
        context_idxs = torch.tensor([wti[w] for w in context], dtype=torch.long)
        model.zero_grad()
        log_probs = model(context_idxs)
        
        loss = loss_fn(log_probs, torch.tensor([wti[target]], dtype=torch.long))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print('Epoch {}: loss={:.4f}'.format(epoch, total_loss))


# Generate some text
context = ['the', 'very']
for i in range(10):
    context_idxs = torch.tensor([wti[w] for w in context], dtype=torch.long)
    log_probs = model(context_idxs)
    next_word_idx = torch.argmax(log_probs).item()
    next_word = text[next_word_idx]
    print(' '.join(context) + ' ' + next_word)
    context = context[1:] + [next_word]






