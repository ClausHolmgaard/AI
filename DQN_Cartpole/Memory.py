import numpy as np
import random


class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        
        self.samples = []

    def add(self, sample):
        self.samples.append(sample)        

        if len(self.samples) > self.capacity:
            self.samples.pop(0)
    
    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    def is_memory_full(self):
        return self.added_samples >= self.capacity

if __name__ == "__main__":
    
    m = Memory(5)

    s = a = r = s_ = 1

    for _ in range(4):
        m.add(s, a, r, s_)
        s += 1
        a += 1
        r += 1
        s_ += 1

    print(m.samples)
    #print(m.added_samples)
    #print(m.samples[-3:])
    print()
    print(m.sample(4))
