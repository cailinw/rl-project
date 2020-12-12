import torch
from generator import Generator

gen = Generator(2,[10,20],['a','b'])
#print(gen.generate(1,2,False,False,True))

print(gen.get_hidden_state(['test1','test2']))
