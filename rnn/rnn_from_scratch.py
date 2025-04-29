import random
import math

class RNN:
    def __init__(self, vocab_size, hidden_size, learning_rate=0.05):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        random.seed(42)
        self.Wx = [[random.random() * 0.02 - 0.01 for _ in range(vocab_size)] 
                   for _ in range(hidden_size)]
        self.Wh = [[random.random() * 0.02 - 0.01 for _ in range(hidden_size)] 
                   for _ in range(hidden_size)]
        self.Wy = [[random.random() * 0.02 - 0.01 for _ in range(hidden_size)] 
                   for _ in range(vocab_size)]
    
    def tanh(self, x):
        return math.tanh(x)
    
    def softmax(self, x):
        max_x = max(x)
        exp_x = [math.exp(xi - max_x) for xi in x]
        sum_exp_x = sum(exp_x)
        return [ex / sum_exp_x for ex in exp_x]
    
    def matrix_vector_multiply(self, A, v):
        result = [0.0 for _ in range(len(A))]
        for i in range(len(A)):
            for j in range(len(v)):
                result[i] += A[i][j] * v[j]
        return result
    
    def vector_add(self, a, b):
        return [ai + bi for ai, bi in zip(a, b)]
    
    def forward(self, inputs, h_prev):
        hs, ys = {}, {}
        hs[-1] = h_prev[:]
        for t in range(len(inputs)):
            wx = self.matrix_vector_multiply(self.Wx, inputs[t])
            wh = self.matrix_vector_multiply(self.Wh, hs[t-1])
            h_input = self.vector_add(wx, wh)
            hs[t] = [self.tanh(x) for x in h_input]
            wy = self.matrix_vector_multiply(self.Wy, hs[t])
            ys[t] = self.softmax(wy)
        return ys, hs
    
    def loss(self, ys, targets):
        loss = 0.0
        for t in range(len(targets)):
            target_idx = targets[t].index(1.0)
            prob = ys[t][target_idx]
            loss += -math.log(max(prob, 1e-10))
        return loss
    
    def backward(self, inputs, targets, hs, ys):
        dWx = [[0.0 for _ in range(self.vocab_size)] for _ in range(self.hidden_size)]
        dWh = [[0.0 for _ in range(self.hidden_size)] for _ in range(self.hidden_size)]
        dWy = [[0.0 for _ in range(self.hidden_size)] for _ in range(self.vocab_size)]
        dh_next = [0.0 for _ in range(self.hidden_size)]
        for t in reversed(range(len(inputs))):
            dy = ys[t][:]
            target_idx = targets[t].index(1.0)
            dy[target_idx] -= 1.0
            for i in range(self.vocab_size):
                for j in range(self.hidden_size):
                    dWy[i][j] += dy[i] * hs[t][j]
            dh = [sum(self.Wy[i][j] * dy[i] for i in range(self.vocab_size)) + dh_next[j] 
                  for j in range(self.hidden_size)]
            dh_raw = [dh[j] * (1 - hs[t][j]**2) for j in range(self.hidden_size)]
            for i in range(self.hidden_size):
                for j in range(self.vocab_size):
                    dWx[i][j] += dh_raw[i] * inputs[t][j]
                for j in range(self.hidden_size):
                    dWh[i][j] += dh_raw[i] * hs[t-1][j]
            dh_next = self.matrix_vector_multiply(
                [[self.Wh[i][j] for j in range(self.hidden_size)] 
                 for i in range(self.hidden_size)], 
                dh_raw
            )
        for dparam in [dWx, dWh, dWy]:
            for i in range(len(dparam)):
                for j in range(len(dparam[i])):
                    dparam[i][j] = max(min(dparam[i][j], 5.0), -5.0)
        return dWx, dWh, dWy
    
    def update_weights(self, dWx, dWh, dWy):
        for i in range(self.hidden_size):
            for j in range(self.vocab_size):
                self.Wx[i][j] -= self.learning_rate * dWx[i][j]
        for i in range(self.hidden_size):
            for j in range(self.hidden_size):
                self.Wh[i][j] -= self.learning_rate * dWh[i][j]
        for i in range(self.vocab_size):
            for j in range(self.hidden_size):
                self.Wy[i][j] -= self.learning_rate * dWy[i][j]
    
    def train(self, inputs, targets, vocab, epochs=2000):
        h_prev = [0.0 for _ in range(self.hidden_size)]
        word_list = list(vocab.keys())
        print(f"Sequence: {' '.join(word_list[:3])} -> {word_list[3]}")
        print(f"{'Epoch':<8} {'Loss':<8} {'Predicted':<12} {'Probabilities'}")
        print("-" * 60)
        for epoch in range(epochs):
            ys, hs = self.forward(inputs, h_prev)
            loss = self.loss(ys, targets)
            dWx, dWh, dWy = self.backward(inputs, targets, hs, ys)
            self.update_weights(dWx, dWh, dWy)
            if epoch % 100 == 0:
                prediction = ys[len(inputs)-1]
                predicted_word_idx = prediction.index(max(prediction))
                predicted_word = word_list[predicted_word_idx]
                prob_str = ", ".join(f"{word}: {prob:.4f}" for word, prob in zip(word_list, prediction))
                print(f"{epoch:<8} {loss:.4f}   {predicted_word:<12} {prob_str}")
        print("-" * 60)
        return ys
    
    def predict(self, inputs):
        h_prev = [0.0 for _ in range(self.hidden_size)]
        ys, _ = self.forward(inputs, h_prev)
        return ys[len(inputs)-1]

def prepare_data():
    words = ["sun", "rises", "every", "morning"]
    vocab = {word: idx for idx, word in enumerate(words)}
    def one_hot(word):
        vec = [0.0] * len(vocab)
        vec[vocab[word]] = 1.0
        return vec
    inputs = [one_hot("sun"), one_hot("rises"), one_hot("every")]
    targets = [one_hot("rises"), one_hot("every"), one_hot("morning")]
    return inputs, targets, vocab

def main():
    vocab_size = 4
    hidden_size = 5
    learning_rate = 0.05
    inputs, targets, vocab = prepare_data()
    rnn = RNN(vocab_size, hidden_size, learning_rate)
    final_outputs = rnn.train(inputs, targets, vocab, epochs=2000)
    prediction = rnn.predict(inputs)
    word_list = list(vocab.keys())
    predicted_word_idx = prediction.index(max(prediction))
    predicted_word = word_list[predicted_word_idx]
    predicted_prob = prediction[predicted_word_idx]
    print("\nFinal Prediction:")
    print(f"Predicted 4th word: {predicted_word} (Probability: {predicted_prob:.4f})")
    print(f"True 4th word: morning")
    print(f"Prediction {'correct' if predicted_word == 'morning' else 'incorrect'}")
    print("Probability distribution:")
    for word, prob in zip(word_list, prediction):
        print(f"  {word}: {prob:.4f}")
    print("-" * 60)

if __name__ == "__main__":
    main()
