
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

local function show_figure(list_color, list, truth, Name,corr)

	Truth=torch.Tensor(#truth)
	Output=torch.Tensor(#list)
	Output_color=torch.Tensor(#list)
	local coef=1
	if corr<0 then coef=-1 end
	for i=1, #truth do
			Truth[i]=truth[i]
			Output[i]=list[i]*coef
			Output_color[i]=list_color[i]*coef
	end
	Truth=(Truth-Truth:mean())/Truth:std()
	Output=(Output-Output:mean())/Output:std()
	Output_color=(Output_color-Output_color:mean())/Output_color:std()

	local point=point or '-'
	-- log results to files
	accLogger = optim.Logger(Name)

	for i=1, #list do
	-- update logger
		accLogger:add{["State-DA"] = Output_color[i],["State"] = Output[i],
				["Ground-truth"] = Truth[i]}
	end
	-- plot logger
	accLogger:style{["State-DA"] = point,["State"] = point,["Ground-truth"] = point}
	accLogger.showPlot = false
	accLogger:plot()
end

local function get_Activation(img,net,level,list_out)
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

local function save_Image_color(net,path,imgs,imgs_color)
	nbIm=#imgs
	local list_color, list_out={},{}
	for i=1, nbIm do
		tensor, list_out=get_Activation(imgs[i],net,19,list_out)
		tensor_color, list_color=get_Activation(imgs_color[i],net,19,list_color)
		out=torch.cat(tensor,tensor_color,2)
		filename=path.."image"..i..".jpg"
		image.save(filename,out)
		xlua.progress(i,nbIm)
	end

	return list_color, list_out
end

local function get_output(net,imgs,imgs_color)
	nbIm=#imgs
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
name_deep='./Log/3_11_reload_2114/Everything/Save3_11_reload_2114_best.t7'
--name_deep='./Log/3_11_reload_211woda/Everything/Save3_11_reload_211woda_best.t7'

name_deep_WODA='./Log/2_11_WODA/Everything/Save2_11_WODA.t7'
name_linear='./Log/2_11_1L_WODA/Everything/Save2_11_1L_WODA_best.t7'

local net = torch.load(name_deep):double()
local WODA = torch.load(name_deep_WODA):double()
local linear= torch.load(name_linear):double()

print('net\n' .. net:__tostring());
local path= paths.home.."/Bureau/Resultat_non_supervise/04-11-2/"

cutorch.setDevice(2) 

local use_simulate_images=true
local list_folders_images, list_txt=Get_HeadCamera_HeadMvt(use_simulate_images)
local last_indice=#list_folders_images
local list=images_Paths(list_folders_images[last_indice])
local imgs=load_list(list, 200, 200, false)
local imgs_color=load_list(list, 200, 200, true)


list_color, list_out = save_Image_color(net,path.."net_",imgs,imgs_color)
list_color_woda, list_out_woda = save_Image_color(WODA,path.."woda_",imgs,imgs_color)
--list_color, list_out=get_output(net,imgs,imgs_color)
--list_color_woda, list_out_woda = get_output(WODA,imgs,imgs_color)
list_linear_color, list_linear=get_output(linear,imgs,imgs_color)

local truth=getTruth(list_txt[last_indice],true)

print("color woda")
ComputeCorrelation(list_color_woda, truth,1)
print("without color woda")
corr=ComputeCorrelation(list_out_woda, truth,1)
show_figure(list_color_woda, list_out_woda, truth,path.."res_woda.log",corr)

print("color")
ComputeCorrelation(list_color, truth,1)
print("without color")
corr=ComputeCorrelation(list_out, truth,1)
show_figure(list_color, list_out, truth,path.."res.log",corr)


print("color linear")
ComputeCorrelation(list_linear_color, truth,1)
print("without color linear")
corr=ComputeCorrelation(list_linear, truth,1)
show_figure(list_linear_color, list_linear, truth,path.."res_linear.log",corr)


