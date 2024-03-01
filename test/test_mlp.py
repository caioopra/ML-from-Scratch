from src.models import MLP
from src.nn import tanh
from src.Value import Value


def test_mlp():
    n = MLP(3, [4, 4, 1], activation=tanh)

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]

    ys = [1.0, -1.0, -1.0, 1.0]
    
    for k in range(100):
        y_pred = [n(x) for x in xs]
        loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, y_pred)], Value(0.0))
        
        for p in n.parameters():
            p.grad = 0.0
            
        loss.backward()
        
        for p in n.parameters():
            p.data -= 0.05 * p.grad
        
        if k % 10 == 0:
            print(k, loss.data)
            
    print(y_pred)