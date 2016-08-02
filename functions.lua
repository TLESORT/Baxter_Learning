
function save_model(model,path)
	print("Saved at : "..path)
	model:cuda()
	parameters, gradParameters = model:getParameters()
	local lightModel = model:clone('weight','bias','running_mean','running_std'):double()
	torch.save(path,model)
end

function preprocessing(im, lenght, width, SpacialNormalization)

	local SpacialNormalization= SpacialNormalization or true
		-- Name channels for convenience
	local channels = {'y','u','v'}
	local mean = {}
	local std = {}
	data = torch.Tensor( 3, lenght, width)
	data:copy(im)
	for i,channel in ipairs(channels) do
	   -- normalize each channel globally:
	   mean[i] = data[i]:mean()
	   std[i] = data[{i,{},{}}]:std()
	   data[{i,{},{}}]:add(-mean[i])
	   data[{i,{},{}}]:div(std[i])
	end

	--preprocessing data: normalize all three channels locally----------------
	if SpacialNormalization then
		-- Define the normalization neighborhood:
		local neighborhood = image.gaussian1D(5) -- 5 for face detector training

		-- Define our local normalization operator
		local normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1e-4)

		-- Normalize all channels locally:
		for c in ipairs(channels) do
		      data[{{c},{},{} }] = normalization:forward(data[{{c},{},{} }])
		end
	end
	return data
end

-- this function search the indice of associated images and take the corresponding images in imgs which are the loaded images of the folder
function getBatch(imgs, list, indice, lenght, width, height, Type)
	
	if (indice+1)*lenght<#list.im1 then
		start=indice*lenght
	else
		start=#list.im1-lenght
	end
	if Type=="Prop" then
		Batch=torch.Tensor(4, lenght,3, width, height)
	else
		Batch=torch.Tensor(2, lenght,3, width, height)
	end
	
	for i=1, lenght do
		Batch[1][i]=imgs[list.im1[start+i]]
		Batch[2][i]=imgs[list.im2[start+i]]
		if Type=="Prop" then
			Batch[3][i]=imgs[list.im3[start+i]]
			Batch[4][i]=imgs[list.im4[start+i]]
		end
	end

	return Batch

end

function Print_performance(Model,imgs, epoch)
		local list_out1={}
		for i=1, #imgs do
			image1=imgs[i]
			local Data1=image1:cuda()
			ForthD= nn.Sequential()
			ForthD:add(nn.Unsqueeze(1,3))
			ForthD=ForthD:cuda()
			Data1=ForthD:forward(Data1)
			Model:forward(Data1)
			local State1=Model.output[1]		
				
			table.insert(list_out1,State1)
		end
		show_figure(list_out1, './Log/state'..epoch..'.log', 1000)
end

function Print_Loss(Temp_loss_list,Prop_loss_list,Rep_loss_list)
	show_figure(Temp_loss_list, './Log/Temp_loss.log', 1000)
	show_figure(Prop_loss_list,  './Log/Prop_loss.log', 1000)
	show_figure(Rep_loss_list, './Log/Rep_loss.log', 1000)
end

function load_list(list,lenght,height)
	im={}
	lenght=lenght or 200
	height=height or 200
	for i=1, #list do
		table.insert(im,getImage(list[i],lenght,height, false))
	end 
	return im
end

function getImage(im,length,height,SpacialNormalization)
	if im=='' or im==nil then return nil end
	local image1=image.load(im,3,'byte')
	local format=length.."x"..height
	local img1_rsz=image.scale(image1,format)
	return preprocessing(img1_rsz,length,height, SpacialNormalization)
end

function show_figure(list_out1, Name , scale)

	local scale=scale or 1000
	-- log results to files
	accLogger = optim.Logger(Name)

	for i=1, #list_out1 do
	-- update logger
		accLogger:add{['out1'] = list_out1[i]*scale}
	end
	-- plot logger
	accLogger:style{['out1'] = '+'}
	accLogger:plot()
end

---------------------------------------------------------------------------------------
-- Function : shuffleDataList(im_list)
-- Input (im_list): list to shuffle
-- Output : The previous list after shuffling
---------------------------------------------------------------------------------------
function shuffleDataList(im_list)
	local rand = math.random 
	local iterations = #im_list.Mode
	local j

	for i = iterations, 2, -1 do
		j = rand(i)
		im_list.im1[i], im_list.im1[j] = im_list.im1[j], im_list.im1[i]
		im_list.im2[i], im_list.im2[j] = im_list.im2[j], im_list.im2[i]
		im_list.im3[i], im_list.im3[j] = im_list.im3[j], im_list.im3[i]
		im_list.im4[i], im_list.im4[j] = im_list.im4[j], im_list.im4[i]
		im_list.Mode[i], im_list.Mode[j]=im_list.Mode[j],im_list.Mode[i]
	end
	return im_list
end
