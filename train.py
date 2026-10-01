import sys 

from model import Model

embed_n = 0
embed_d = 0
hidden_d = 0
win_d = 0 

def _handle_args(): 
    for arg in sys.argv[1:]: 
        if arg.endswith('.py'):
            print(f'overriding config with: {arg}')
            with open(arg) as f: print(f.read())
            exec(open(arg).read(), globals())
    
if __name__ == '__main__': 
    _handle_args()
    model = Model(embed_n, embed_d, hidden_d, win_d)


