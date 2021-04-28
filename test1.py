import json, torch, io, time
from encoder import UniversalEncoder

data = [
    'What color is chameleon?',
    'When is the festival of colors?',
    'When is the next music festival?',
    'How far is the moon?',
    'How far is the sun?',
    'What happens when the sun goes down?',
    'What we do in the shadows?',
    'What is the meaning of all this?',
    'What is the meaning of Russel\'s paradox?',
    'How are you doing?'
]

encoder  = UniversalEncoder("tstsv.ddns.net",8501)
vectors = encoder.encode(data)
''' file = open("test.json","a")
file.write("\n")
file.close() '''
json.dump(vectors, open("test.json","a"))
file = open("test.json","r",buffering=2)
for line in file:
    data = json.load(io.StringIO(line))
    for key, value in data.items():
        print((torch.Tensor(value)).shape)
    print("**************************")

#converter
def load_index(dir:str):
    file = open("test.json","r",buffering=1)
    vectors = {}
    for line in file:
        vectors.update(json.load(io.StringIO(line)))
    vectors.pop(-1)
    return vectors

def save_index(vectors,dir:str):
    file = open("test.json","a")
    json.dump(vectors,file)
    file.write("\n")