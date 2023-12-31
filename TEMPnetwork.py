class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        samples = len(input_data)
        result = []

        #RUN NETWORK OVER ALL SAMPLES
        for i in range(samples):
            #FORWARD
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)
        return result
    
    #TRAIN NETWORK
    def fit(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)

        #TRAINING LOOP
        for i in range(epochs):
            err = 0
            for j in range(samples):
                #FORWARD
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward(output)

                #compute loss for display
                err += self.loss(y_train[j], output)   

                #BACKWARD
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)

            #calc avg error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))

        
