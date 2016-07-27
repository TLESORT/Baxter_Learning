
require 'nn'
require 'nngraph'
require 'torch'
require 'xlua'
require 'cunn'


State1 = torch.rand(5,1)
State2 = torch.rand(5,1)
State3 = torch.rand(5,1)
State4 = torch.rand(5,1)

State1 = State1:cuda()
State2 = State2:cuda()
State3 = State3:cuda()
State4 = State4:cuda()

print(State1)
print(State2)

--------------------------------------------------------------------------------------

h1 = nn.Identity()()
h2 = nn.Identity()()
h3 = nn.Identity()()
h4 = nn.Identity()()

h_h1 = nn.CSubTable()({h2,h1})
h_h2 = nn.CSubTable()({h4,h3})

madd = nn.CSubTable()({h_h2,h_h1})
sqr=nn.Square()(madd)
out1 = nn.Sum(1,1)(sqr)

norm2= nn.Sum(1,1)(nn.Square()(nn.CSubTable()({h3,h1})))
out2=nn.Exp()(nn.MulConstant(-1)(norm2))

outTot=nn.Sum(1,1)(nn.CMulTable()({out1, out2}))


gmod = nn.gModule({h1, h2, h3, h4}, {outTot})
gmod=gmod:cuda()

output=gmod:updateOutput({State1, State2, State3, State4})
gmod:updateGradInput({State1, State2, State3, State4}, torch.ones(1))
--------------------------------------------------------------------------------------

print("Premier terme")
print(h_h1.data.module.output)
print(h_h2.data.module.output)
print(madd.data.module.output)
print(sqr.data.module.output)
print(out1.data.module.output)

print("2eme terme")
print(norm2.data.module.output)
print(out2.data.module.output)

print("Produit")
print(output)
--graph.dot(gmod.fg, 'Big MLP')

print(gmod.gradInput[1])
