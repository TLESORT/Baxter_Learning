

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


local function clampImage(tensor)
   if tensor:type() == 'torch.ByteTensor' then
      return tensor
   end
    
   local a = torch.Tensor():resize(tensor:size()):copy(tensor)
   min=a:min()
   max=a:max()
   a:add(-min)
   a:mul(1/(max-min))         -- remap to [0-1]
   return a
end

function pcacov(x)
   local mean = torch.mean(x,1)
   local xm = x - mean:expandAs(x)
   local c = torch.mm(xm:t(),xm)
   c:div(x:size(1)-1)
   local ce,cv = torch.symeig(c,'V')
   return ce,cv
end


function gamma(im)
	local Gamma= torch.Tensor(3,3)
	local channels = {'y','u','v'}
	local mean = {}
	local std = {}
	for i,channel in ipairs(channels) do
		
		for j,channel in ipairs(channels) do
	   		if i==j then Gamma[i][i] = im[{i,{},{}}]:var()
			else
				chan_i=im[{i,{},{}}]-im[{i,{},{}}]:mean()
				chan_j=im[{j,{},{}}]-im[{j,{},{}}]:mean()
				Gamma[i][j]=(chan_i:t()*chan_j):mean()
			end
		end
	end

	return Gamma
end

function transformation(im, v,e,Gaus)
	transfo=torch.Tensor(3,200,200)
	for i=1, 200 do
		for j=1, 200 do
			transfo[{{},i,j}]=im[{{},i,j}]+(torch.mv(v,e))*Gaus[i][j]
		end
	end
 return transfo
end

function loi_normal(x,y,center_x,center_y,std_x,std_y)

 return math.exp(-(x-center_x)^2/(2*std_x^2))*math.exp(-(y-center_y)^2/(2*std_y^2))

end

local use_simulate_images=true
local list_folders_images, list_txt=Get_HeadCamera_HeadMvt(use_simulate_images)
local last_indice=#list_folders_images
local list=images_Paths(list_folders_images[last_indice])

local nbIm=#list
local imgs=load_list(list, 200, 200)

im=imgs[1]
image.display{im}
gam=gamma(im)
e, V = torch.eig(gam,'V')

print(gam)
--print(V*torch.diag(e:select(2, 1))*V:t())

factors=torch.randn(3)*0.1
--print("factors")
--print(factors)
print(e)
for i=1,3 do e:select(2, 1)[i]=e:select(2, 1)[i]*factors[i] end
print(e)
print(V*torch.diag(e:select(2, 1))*V:t())
--gam=gam+V*torch.diag(e:select(2, 1))*V:t()

print(gam)

--transfo=transformation(im, gam)



Gaus=torch.zeros(200,200)
foyer_x=torch.random(1,200)
foyer_y=torch.random(1,200)	
std_x=torch.random(1,5)
std_y=torch.random(1,5)
--[[
for i=1, 100000 do
	a=torch.normal(foyer_x,std_x)
	b=torch.normal(foyer_y,std_y)
	if not(a>199 or b>199 or a<2 or b<2) then 
	Gaus[torch.floor(a)][torch.floor(b)]=Gaus[torch.floor(a)][torch.floor(b)]+5
	Gaus[torch.floor(a)+1][torch.floor(b)]=Gaus[torch.floor(a)+1][torch.floor(b)]+5
	Gaus[torch.floor(a)][torch.floor(b)+1]=Gaus[torch.floor(a)][torch.floor(b)+1]+5
	Gaus[torch.floor(a)-1][torch.floor(b)]=Gaus[torch.floor(a)-1][torch.floor(b)]+5
	Gaus[torch.floor(a)][torch.floor(b)-1]=Gaus[torch.floor(a)][torch.floor(b)-1]+5
	Gaus[torch.floor(a)+1][torch.floor(b)-1]=Gaus[torch.floor(a)+1][torch.floor(b)-1]+5
	Gaus[torch.floor(a)+1][torch.floor(b)+1]=Gaus[torch.floor(a)+1][torch.floor(b)+1]+5
	Gaus[torch.floor(a)-1][torch.floor(b)+1]=Gaus[torch.floor(a)-1][torch.floor(b)+1]+5
	Gaus[torch.floor(a)-1][torch.floor(b)-1]=Gaus[torch.floor(a)-1][torch.floor(b)-1]+5
	end
end
--]]
GAUS=torch.zeros(3,200,200)

for x=1,200 do
for y=1,200 do
	Gaus[x][y]=loi_normal(x/20,y/20,foyer_x/20,foyer_y/20,std_x,std_y)
end
end
image.display(Gaus)
GAUS[1]=Gaus
GAUS[2]=Gaus
GAUS[3]=Gaus
--color=transformation(GAUS*-1, V,e:select(2, 1))
--color=clampImage(color)
--image.display(color)
transfo=transformation(im, V,e:select(2, 1),Gaus)
transfo=clampImage(transfo)
im=clampImage(im)

image.display{image={im, transfo}}


