import pickle
from generator import Generator

token_map = pickle.load(open('save/vocab_map.pkl', 'rb'))

gen = Generator(10,token_map)

print(gen.generate(2,1,False,False,False))
