
require 'nn'
require 'optim'
require 'image'
require 'torch'
require 'xlua'
require 'math'
require 'string'
require 'cunn'
require 'nngraph'

require 'MSDC'
require 'functions.lua'
require "Get_HeadCamera_HeadMvt"
require 'priors'


-- Load le modèle entrainer
net = torch.load('./Save/Save03_08_real.t7'):double()
print('net\n' .. net:__tostring());

--recuperer une image
local use_simulate_images=true
local list_folders_images, list_txt=Get_HeadCamera_HeadMvt(use_simulate_images)
local list=images_Paths(list_folders_images[1])
imgs=load_list(list, 200, 200)

im=imgs[1]
Batch=torch.Tensor(1,3, 200, 200) --nécéssité d'un batch pour le fonctionnement
Batch[1]=im


mean=im:mean()
std=im:std()

Layer=17
FM=1
LR=100


-- creation reseau pour la descente de gradient sur une seule feature map

new=nn.Sequential()
for i=1, Layer do
	new:add(net:get(i))
end

new:add(nn.SplitTable(1)):add(nn.SelectTable(FM))

print(new:__tostring());

-- calculer la feature map pour l'image

new:forward(Batch)

label=torch.Tensor(new.output:size())
label=new.output:clone()

--label=torch.zeros(1,10,10)
--label[1][6][6]=10

-- générer une image bruit blanc
noise=torch.rand(1,3,200,200)


--on fait en sorte que l'image d'entrée et le bruit est la meme moyenne et variance
noise[1]:add(-(noise[1]:mean())+mean)
noise[1]:div(noise[1]:std()):mul(std)

new:forward(noise)
res=new.output
image.display({image={label,res}, zoom=10})

--passer l'image a travers le reseau
criterion= nn.MSECriterion()

image.display(im)
image.display(noise[1])
for i=1,100 do
	new:forward(noise)
	res=new.output
	criterion:forward(res,label)
	grad=criterion:backward(res,label)
	new:backward(noise,grad)
	noise=noise-LR*new:get(1).gradInput
end
image.display({image={label,res}, zoom=10})
image.display(noise[1])

--faire un descente de gradient pour rendre les deux features map identique

-- visualiser
