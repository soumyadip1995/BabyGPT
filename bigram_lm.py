import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple vocabulary of words
with open(r"C:\Users\Soumyadip Nandi\Downloads\policy\language\text.txt", 'r', encoding='utf-8') as f:
    vocab = f.read().split()

# Create a dictionary to map each word to a unique index
word2idx = {word: idx for idx, word in enumerate(vocab)}

encode = lambda s: [word2idx[c] for c in s]


# Define a function to convert a list of words to a list of indices
def words_to_indices(words, word2idx):
    return [word2idx[word] for word in words]

# Define a bigram language model using PyTorch
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(BigramLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.linear1 = nn.Linear(embedding_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs):
        embeds = self.embedding(inputs)
        hidden = torch.relu(self.linear1(embeds))
        outputs = self.linear2(hidden)
        return outputs

# Create a bigram language model instance
model = BigramLanguageModel(len(vocab), 10, 20)

# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
epoch = 1000

# Define a sample sequence of words
words =  ['The', 'very', 'next', 'day']

# Convert the sequence of words to a sequence of indices
inputs = torch.LongTensor(words_to_indices(words, word2idx))

# Train the bigram language model
for epoch in range(epoch):
    optimizer.zero_grad()
    outputs = model(inputs[:-1])
    targets = inputs[1:] # first coloumn
    loss = criterion(outputs.view(-1, len(vocab)), targets)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Generate a sequence of words using the bigram language model
current_word = 'the'
generated_words = [current_word]
for i in range(10):
    inputs = torch.LongTensor([word2idx[current_word]])
    outputs = model(inputs)
    _, predicted = torch.max(outputs, dim=1)
    current_word = vocab[predicted.item()]
    generated_words.append(current_word)
print(' '.join(generated_words))
