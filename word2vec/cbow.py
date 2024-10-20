import numpy as np

class CBOW:
    def __init__(self, vocab_size, embedding_dim, learning_rate):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        
      
        self.W1 = np.random.randn(vocab_size, embedding_dim)  # W_context
        self.W2 = np.random.randn(embedding_dim, vocab_size)  # W_word
    
    def one_hot_vector(self, word, word2idx):
        """Convert a word into one-hot encoding."""
        one_hot = np.zeros(self.vocab_size)
        one_hot[word2idx[word]] = 1
        return one_hot
    
    def softmax(self, x):
        """Softmax function to convert raw scores into probabilities."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=0)
    
    def forward(self, context_words, W1, W2):
        """Perform forward propagation: Compute the prediction from context."""
       
        avg_context_vecs=np.mean(context_words,axis=0)
        print(avg_context_vecs.shape,'avg')
        #(12,) @ 12 x 10
        h=avg_context_vecs@W1
        print('j',h.shape) #10
       

        print('W2',W2.shape) # 10,12
        u = np.dot(W2.T, h)  # Output layer
        print('u',u.shape) #12
        y_pred = self.softmax(u)
        return y_pred, h, avg_context_vecs
    
    def compute_loss(self, y_true, y_pred):
        """Cross-entropy loss function."""
        return -np.sum(y_true * np.log(y_pred))
    
    def train(self, corpus, word2idx, idx2word, context_window=2, epochs=5000):
        """Train the CBOW model on the given corpus."""
        for epoch in range(epochs):
            total_loss = 0
            
            for sentence in corpus:
                words = sentence.lower().split()
                for i, target_word in enumerate(words):
                    # Create context window
                    context_words = []
                    for j in range(max(0, i - context_window), min(len(words), i + context_window + 1)):
                        if i != j:
                            context_words.append(words[j])
                    
                    # One-hot encode context and target
                    context_vecs = np.array([self.one_hot_vector(w, word2idx) for w in context_words])
                    target_vec = self.one_hot_vector(target_word, word2idx)
                    
                  
                    
                    y_pred, h,average_context_vecs = self.forward(context_vecs, self.W1, self.W2)
                    
                    # Compute loss
                    loss = self.compute_loss(target_vec, y_pred)
                    total_loss += loss
                    
       
                    e = y_pred - target_vec  # Error between prediction and target (12,)
                    
                    dW2 = np.outer(h, e)  # Gradient for W2 #(10,12)
                    dh= np.dot(self.W2,e)#(10,)
                    # print('dh',dh.shape)
                    print('W!',self.W1.shape)
                    dW1= np.outer(average_context_vecs,dh)  #(12,10)
                    print('dW1',dW1.shape)
                  
          
                    # Update weights
                    self.W1 -= self.learning_rate * dW1  # Ensure shapes match (12, 10)
                    self.W2 -= self.learning_rate * dW2  # Shape (10, 12)
                    self.W2 -= self.learning_rate * dW2
            
            if epoch % 500 == 0:
                print(f"Epoch: {epoch}, Loss: {total_loss}")
    
    def predict(self, context_words, word2idx, idx2word):
        """Predict the target word given context words."""
        context_vecs = np.array([self.one_hot_vector(w, word2idx) for w in context_words])
        y_pred, _ ,avg_context_words= self.forward(context_vecs, self.W1, self.W2)
        predicted_word = idx2word[np.argmax(y_pred)]
        return predicted_word


# Sample corpus
corpus = [
    "the cat sits on the mat",
    "the dog lays on the rug",
    "the cat chases the mouse",
    "the dog barks at the cat"
]

# Preprocessing: Tokenize and build vocabulary
def tokenize(corpus):
    tokens = []
    for sentence in corpus:
        tokens.extend(sentence.lower().split())
    return tokens

def build_vocab(tokens):
    vocab = list(set(tokens))
    word2idx = {word: i for i, word in enumerate(vocab)}
    idx2word = {i: word for i, word in enumerate(vocab)}
    return vocab, word2idx, idx2word

tokens = tokenize(corpus)
vocab, word2idx, idx2word = build_vocab(tokens)
vocab_size = len(vocab)


embedding_dim = 10
learning_rate = 0.01
epochs = 5000
context_window = 2


cbow_model = CBOW(vocab_size, embedding_dim, learning_rate)


cbow_model.train(corpus, word2idx, idx2word, context_window, epochs)


context_words = ["the", "dog", "on", "the"]
predicted_word = cbow_model.predict(context_words, word2idx, idx2word)
print(f"Predicted word for the context {context_words}: {predicted_word}")