import pickle
from generator import Generator

str_map = pickle.load(open('save/str_map.pkl', 'rb'))

gen = Generator(10,str_map)

print(gen.generate(2,1,torch.arange(0,16).view(2,2,4), False,False,False))
