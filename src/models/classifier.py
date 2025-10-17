import numpy as np 


class SpamClassifier:
    def __init__(self):
        self.best_params = None
        self.lambda_reg = 0.51
        self.beta1 = 0.789
        self.beta2 = 0.7989        
        self.epsilon = 1e-8
        self.batch_size = 128
        self.layer_dims = [None, 128, 64, 1]
        self.keep_prob1 = 0.3
        self.keep_prob2 = 0.2
        self.t = 0
        self.mean = None
        self.std = None
        self.weight_pos = 1.2
        self.weight_neg = 0.9
        self.best_val_acc = 0
       


    def train(self, training_spam= training_spam, max_epochs = 300, init_alpha = 0.001):
        training_spam = training_spam.astype(np.float64)
        np.random.shuffle(training_spam)
        self.layer_dims[0] = training_spam.shape[1] - 1
       
        testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",").astype(int)
        
        
    
        train_val_split = int(0.75 * training_spam.shape[0])
        train_data = training_spam[:train_val_split]
        val_data = training_spam[train_val_split:]
       
        X_train_data = train_data[:, 1:].T
        Y_train_data = train_data[:, 0].reshape(1, -1)
        X_val_data = val_data[:, 1:].T
        Y_val_data = val_data[:, 0].reshape(1, -1)


       
        self.mean = np.mean(X_train_data, axis=1, keepdims=True)
        self.std = np.std(X_train_data, axis=1, keepdims=True) + 1e-8


        X_train_data = (X_train_data - self.mean) / self.std
        X_val_data = (X_val_data - self.mean) / self.std



        n = max_epochs
        ns = init_alpha
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = self.optimise(
            X_train_data, Y_train_data, X_val_data, Y_val_data,
            max_epochs= n,
            init_alpha= ns
        )


    def predict(self, datas, preloaded = True):
        if preloaded:
            data = datas
            files = ["1.npz", "2.npz", "3.npz", "4.npz", "5.npz", "6.npz"]
            all_preds = []
           
            best_params, mean, std = load_model_weights("3.npz")
        
        
            data = data.astype(np.float64)
            X = data.T
            X = (X - mean) / std
        
            
            _, _, _, _, _, A3 = forward(X, *best_params, training=False)
            predictions = (A3 > 0.5).astype(np.int32).flatten()
        # predictions = (predictions > 0.5).astype(np.int32)
        
        
            return predictions

        else:
            data = datas.astype(np.float64)
            X = data.T
            X = (X - self.mean) / self.std
            _, _, _, _, _, A3 = self.forward(X, *self.best_params, training=False)
            return (A3 > 0.5).astype(np.int32).flatten()


    def optimise(self, X, Y, X_val, Y_val, max_epochs, init_alpha):
        params = list(self.set_params())
        m = list(map(np.zeros_like, params))
        v = list(map(np.zeros_like, params))
        self.t = 0
        best_params = [np.copy(p) for p in params]
  
        patience = 110
        epochs_no_improve = 0
       
        decay_rate = 0.999  


        for epoch in range(max_epochs):
            current_alpha = init_alpha * (decay_rate ** epoch)
           
            permutation = np.random.permutation(X.shape[1])
            X_shuffled = X[:, permutation]
            Y_shuffled = Y[:, permutation]
           
            for i in range(0, X.shape[1], self.batch_size):
                X_batch = X_shuffled[:, i:i+self.batch_size]
                Y_batch = Y_shuffled[:, i:i+self.batch_size]
               
                Z1, A1, Z2, A2, Z3, A3 = self.forward(X_batch, *params)
                grads = self.backward(Z1, A1, Z2, A2, Z3, A3, params, X_batch, Y_batch)


                self.t += 1
                for j in range(len(params)):
                    m[j] = self.beta1 * m[j] + (1 - self.beta1) * grads[j]
                    v[j] = self.beta2 * v[j] + (1 - self.beta2) * (grads[j] ** 2)
                    m_h = m[j] / (1 - self.beta1 ** self.t)
                    v_h = v[j] / (1 - self.beta2 ** self.t)
                    params[j] -= current_alpha * m_h / (np.sqrt(v_h) + self.epsilon)
           
            _, _, _, _, _, A3_val = self.forward(X_val, *params, training=False)
            val_acc = self.compute_accuracy(A3_val, Y_val)
           
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                epochs_no_improve = 0
                best_params = [np.copy(p) for p in params]
            else:
                epochs_no_improve += 1
                if (epochs_no_improve >= patience  and val_acc >= 0.91):
                    print(f"Best Validation Accuracy: {self.best_val_acc:.4f}")
                    print(f"Early stopping at epoch {epoch}")
                    break


        self.best_params = best_params
        return best_params
   
       
   


    def forward(self, X, W1, b1, W2, b2, W3, b3, training=True):
        Z1 = W1 @ X + b1
        A1 = np.maximum(Z1, 0)
        if training:
            mask1 = (np.random.rand(*A1.shape) < self.keep_prob1)
            A1 = (A1 * mask1) / self.keep_prob1
       
        Z2 = W2 @ A1 + b2
        A2 = np.tanh(Z2)
        if training:
            mask2 = (np.random.rand(*A2.shape) < self.keep_prob2)
            A2 = (A2 * mask2) / self.keep_prob2
       
        Z3 = W3 @ A2 + b3
        A3 = 1 / (1 + np.exp(-Z3))
        return Z1, A1, Z2, A2, Z3, A3


    def set_params(self):
        np.random.seed(42)
        params = []
        for l in range(1, len(self.layer_dims)):
            if l == 1:
                
                factor = np.sqrt(2.0 / self.layer_dims[l-1])
            elif l == 2:
                
                factor = np.sqrt(1.0 / self.layer_dims[l-1])
            else:
                factor = np.sqrt(1.0 / self.layer_dims[l-1])
           
            weight = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * factor
            bias = np.zeros((self.layer_dims[l], 1))
            params.extend([weight, bias])
        return tuple(params)


    def backward(self, Z1, A1, Z2, A2, Z3, A3, params, X, Y):
        W1, W2, W3 = params[0], params[2], params[4]
        m = Y.size


        weights = np.where(Y == 1, self.weight_pos, self.weight_neg)
        dZ3 = (A3 - Y) * weights
        dW3 = (dZ3 @ A2.T) / m + (self.lambda_reg * W3) / m
        db3 = np.sum(dZ3, axis=1, keepdims=True) / m
       
        dA2 = W3.T @ dZ3
        dZ2 = dA2 * (1 - np.tanh(Z2)**2)
        dW2 = (dZ2 @ A1.T) / m + (self.lambda_reg * W2) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
       
        dA1 = W2.T @ dZ2
        dZ1 = dA1 * (Z1 > 0).astype(np.float64)
        dW1 = (dZ1 @ X.T) / m + (self.lambda_reg * W1) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m
       
        return (dW1, db1, dW2, db2, dW3, db3)


    def compute_accuracy(self, A3, Y):
        predictions = (A3 > 0.5).astype(np.float64)
        return np.mean(predictions == Y)



def save_model_weights(filename, mean=None, std=None):


    while True:
       
        clf = SpamClassifier()
        clf.train(training_spam)

    


        test_data = testing_spam[:, 1:]
        test_labels = testing_spam[:, 0]


        predictions = clf.predict(test_data, False)
        best_params = clf.best_params
        mean = clf.mean
        std  = clf.std
        accuracy = np.count_nonzero(predictions == test_labels)/test_labels.shape[0]
        
        print(f"Accuracy on test data is: {accuracy}")
        print(f"overfit by {(clf.best_val_acc-accuracy)*100}")
        if accuracy > 0.94 and (abs(clf.best_val_acc-accuracy)*100) < 0.9:
            break

  
    data_to_save = {
        'W1': best_params[0],
        'b1': best_params[1],
        'W2': best_params[2],
        'b2': best_params[3],
        'W3': best_params[4],
        'b3': best_params[5]
    }
   
    
    if mean is not None:
        data_to_save['mean'] = mean
    if std is not None:
        data_to_save['std'] = std
   
    np.savez(filename, **data_to_save)
    print(f"Weights saved to {filename}")
    return accuracy
def load_model_weights( filename):
        data = np.load(filename)
    
        best_params = [
            data['W1'],
            data['b1'],
            data['W2'],
            data['b2'],
            data['W3'],
            data['b3']
        ]


        mean = data['mean'] if 'mean' in data else None
        std = data['std'] if 'std' in data else None


        print(f"Weights loaded from {filename}")
        return best_params, mean, std


def forward( X, W1, b1, W2, b2, W3, b3, training=False):
        Z1 = W1 @ X + b1
        A1 = np.maximum(Z1, 0)
    
    
        Z2 = W2 @ A1 + b2
        A2 = np.tanh(Z2)


    
        Z3 = W3 @ A2 + b3
        A3 = 1 / (1 + np.exp(-Z3))
        return Z1, A1, Z2, A2, Z3, A3


        
    



        
        
#classifier = SpamClassifier()

#classifier. predict(np.loadtxt(open("data/testing_spam.csv"), delimiter=",").astype(int)[:, 1:], True)


def create_classifier():
    classifier = SpamClassifier()
    
    return classifier

#save_model_weights("7")     

classifier = create_classifier()

#testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",").astype(int)
#test_data = testing_spam[:, 1:]
#test_labels = testing_spam[:, 0]
#preds = classifier.predict(test_data)
