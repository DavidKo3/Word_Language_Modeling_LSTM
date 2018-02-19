import torch

weights = torch.FloatTensor([1,500, 20, 150, 300])

a0=0
a1=0
a2=0
a3=0
a4=0


for i in range(500000):
        a=torch.multinomial(weights, 1)
        if a[0] == 0:
            a0+=1
        if a[0] == 1:
            a1+=1
        if a[0] == 2:
            a2+=1
        if a[0] == 3:
            a3+=1
        if a[0] == 4:
            a4+=1

print (weights)
print (a0, a1, a2, a3, a4)
