import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop


LEARNING_RATE = 0.00025

class Brain(object):
    def __init__(self, state_count, action_count):
        self.state_count = state_count
        self.action_count = action_count

        self.model = self.create_model()
        self.target_model = self.create_model()
    
    def create_model(self):
        model = Sequential()

        model.add(Dense(activation='relu', input_dim=self.state_count, units=64))
        model.add(Dense(activation='linear', units=self.action_count))

        opt = RMSprop(lr=LEARNING_RATE)
        model.compile(loss='mse', optimizer=opt)

        return model

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def train(self, x, y, epoch=1, verbose=0):
        self.model.fit(x, y, batch_size=64, nb_epoch=epoch, verbose=verbose)
    
    def predict(self, s, target=False):
        if target:
            return self.target_model.predict(s)
        else:
            return self.model.predict(s)
    
    def predict_one(self, s, target=False):
        return self.predict(s.reshape(1, self.state_count), target=target).flatten()
    
if __name__ == "__main__":
    
    b = Brain(2, 3)

    s = np.array([2, 2]).reshape(1, -1)
    print()
    print(b.predict(s))
    print(b.predict_one(s))