
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
require 'printing.lua'
require "Get_HeadCamera_HeadMvt"
require 'priors'

function rescale_10_200(im)
	img_rescale=torch.Tensor(3,200, 200)
	for i=1, 3 do
		for j=1,200 do
			for l=1,200 do
				img_rescale[i][j][l]=im[math.ceil(j/20)][math.ceil(l/20)]
			end
		end
	end

	return img_rescale
end

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

local function show_figure(list_color, list, truth, Name)

	Truth=torch.Tensor(#truth)
	Output=torch.Tensor(#list)
	Output_color=torch.Tensor(#list)
	for i=1, #truth do
			Truth[i]=truth[i]
			Output[i]=list[i]*-1
			Output_color[i]=list_color[i]*-1
	end
	Truth=(Truth-Truth:mean())/Truth:std()
	Output=(Output-Output:mean())/Output:std()
	Output_color=(Output_color-Output_color:mean())/Output_color:std()

	local point=point or '-'
	-- log results to files
	accLogger = optim.Logger(Name)

	for i=1, #list do
	-- update logger
		accLogger:add{["Out-DA"] = Output_color[i],
				["Out"] = Output[i],
				["Ground-truth"] = Truth[i]}
	end
	-- plot logger
	accLogger:style{["Out-DA"] = point,
				["Out"] = point,
				["Ground-truth"] = point}
	accLogger.showPlot = false
	accLogger:plot()
end


--image.display{image=net:get(1).weight[{}][{}][{}][{}], nrow=8,  zoom=100, legend="image"}
--image.display{image=net:get(19).output[1], nrow=16,  zoom=10, legend="image"..i}


local function get_Activation(img,model,level,list_out)
	Batch=torch.Tensor(1,3, 200, 200)
	Batch[1]=img
	out=net:forward(Batch)
	table.insert(list_out,out[1])
	im=net:get(level).output[1]
	img_rsz=clampImage(rescale_10_200(im[1]))
	im=clampImage(Batch[1])
	salience=clampImage(torch.cmul(img_rsz,im))
	return torch.cat(torch.cat(img_rsz,salience,3),im,3), list_out
end

local function save_Image_color(list,net,path)
	nbIm=#list
	local imgs=load_list(list, 200, 200, false)
	local imgs_color=load_list(list, 200, 200, true)
	local list_color, list_out={},{}
	for i=1, nbIm do
		tensor, list=get_Activation(imgs[i],net,19,list)
		tensor_color, list_color=get_Activation(imgs_color[i],net,19,list_color)
		out=torch.cat(tensor,tensor_color,2)
		filename=path.."image"..i..".jpg"
		image.save(filename,out)
		xlua.progress(i,nbIm)
	end

	return list_color, list_out
end

local function get_output(list,net,path)
	nbIm=#list
	local imgs=load_list(list, 200, 200, false)
	local imgs_color=load_list(list, 200, 200, true)
	local list_color, list_out={},{}
	for i=1, nbIm do
		Batch=torch.Tensor(1,3, 200, 200)
		Batch[1]=imgs[i]
		table.insert(list_out,net:forward(Batch)[1])
		Batch[1]=imgs_color[i]
		table.insert(list_color,net:forward(Batch)[1])
		xlua.progress(i,nbIm)
	end

	return list_color, list_out
end


torch.manualSeed(123)
local net = torch.load('./Log/24_10_2/Everything/Save24_10_2.t7'):double()
print('net\n' .. net:__tostring());
local path= paths.home.."/Bureau/Resultat_non_supervise/24_10_2/"

local linear= torch.load('./Log/24_10_1L/Everything/Save24_10_1L.t7'):double()

cutorch.setDevice(2) 

local use_simulate_images=true
local list_folders_images, list_txt=Get_HeadCamera_HeadMvt(use_simulate_images)
local last_indice=#list_folders_images
local list=images_Paths(list_folders_images[last_indice])


--list_color, list_out = save_Image_color(list,net,path,19)
--list_color, list_out=get_output(list,net)
list_linear_color, list_linear=get_output(list,linear)

local truth=getTruth(list_txt[last_indice],true)
--show_figure(list_color, list_out, truth,path.."res.log")
show_figure(list_linear_color, list_linear, truth,path.."res_linear.log")

print("color")
ComputeCorrelation(list_color, truth,1)
print("without color")
ComputeCorrelation(list_out, truth,1)

print("color linear")
ComputeCorrelation(list_linear_color, truth,1)
print("without color linear")
ComputeCorrelation(list_linear, truth,1)


