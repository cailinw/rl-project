import pickle
from generator import Generator

str_map = pickle.load(open('save/str_map.pkl', 'rb'))

gen = Generator(10,str_map)

print(gen.generate(2,1,False,False,False))
