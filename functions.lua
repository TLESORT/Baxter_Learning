---------------------------------------------------------------------------------------
-- Function : 
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function save_model(model,path)
	print("Saved at : "..path)
	model:cuda()
	parameters, gradParameters = model:getParameters()
	local lightModel = model:clone('weight','bias','running_mean','running_std'):double()
	torch.save(path,model)
end

---------------------------------------------------------------------------------------
-- Function : 
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function preprocessing(im, lenght, width, SpacialNormalization)

	local SpacialNormalization= (SpacialNormalization==nil and true) or SpacialNormalization
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

---------------------------------------------------------------------------------------
-- Function : 
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
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

---------------------------------------------------------------------------------------
-- Function : 
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function getRandomBatch(imgs, txt, lenght, width, height, Mode, use_simulate_images)
	
	if Mode=="Prop" or Mode=="Rep" then
		Batch=torch.Tensor(4, lenght,3, width, height)
	else
		Batch=torch.Tensor(2, lenght,3, width, height)
	end
	
	for i=1, lenght do
		if Mode=="Prop" or Mode=="Rep" then
			Set=get_one_random_Prop_Set(txt ,use_simulate_images)
			Batch[1][i]=imgs[Set.im1]
			Batch[2][i]=imgs[Set.im2]
			Batch[3][i]=imgs[Set.im3]
			Batch[4][i]=imgs[Set.im4]
		elseif Mode=="Temp" then
			Set=get_one_random_Temp_Set(#imgs)
			Batch[1][i]=imgs[Set.im1]
			Batch[2][i]=imgs[Set.im2]
		else
			print "getRandomBatch Wrong mode "
		end
	end
	return Batch
end

---------------------------------------------------------------------------------------
-- Function : 
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function getRandomBatchFromSeparateList(imgs1, imgs2, txt1, txt2, lenght, image_width, image_height, Mode, use_simulate_images)

	local width=image_width or 200
	local height=image_height or 200

	if Mode=="Prop" or Mode=="Rep" then
		Batch=torch.Tensor(4, lenght,3, width, height)
	else
		Batch=torch.Tensor(2, lenght,3, width, height)
	end
	
	for i=1, lenght do
		if Mode=="Prop" or Mode=="Rep" then
			Set=get_two_Prop_Pair(txt1, txt2, use_simulate_images)
			Batch[1][i]=imgs1[Set.im1]
			Batch[2][i]=imgs1[Set.im2]
			Batch[3][i]=imgs2[Set.im3]
			Batch[4][i]=imgs2[Set.im4]
		elseif Mode=="Temp" then
			Set=get_one_random_Temp_Set(#imgs1)
			Batch[1][i]=imgs1[Set.im1]
			Batch[2][i]=imgs1[Set.im2]
		else
			print "getRandomBatch Wrong mode "
		end
	end
	return Batch

end

---------------------------------------------------------------------------------------
-- Function : 
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function copy_weight(model, AE)
	model:get(1).weight:copy(AE:get(1).weight)
	model:get(4).weight:copy(AE:get(5).weight)
	return model
end

---------------------------------------------------------------------------------------
-- Function : 
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function Print_performance(Models,imgs,txt, name, Log_Folder, use_simulate_images)

	local REP_criterion=get_Rep_criterion()
	local PROP_criterion=get_Prop_criterion()
	local CAUS_criterion=get_Caus_criterion()
	local TEMP_criterion=nn.MSDCriterion()

	local Temp=0
	local Rep=0
	local Prop=0
	local Model=Models.Model1

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

	-- biased estimation of test loss
	local nb_sample=100
	for i=1, nb_sample do
		Prop_batch=getRandomBatch(imgs, txt, 1, 200, 200, 'Prop', use_simulate_images)
		Temp_batch=getRandomBatch(imgs, txt, 1, 200, 200, 'Temp', use_simulate_images)

		Temp=Temp+doStuff_temp(Models,TEMP_criterion, Temp_batch)
		Prop=Prop+doStuff_Prop(Models,PROP_criterion,Prop_batch)	
		--loss=doStuff_Caus(Models,criterion,batch)
		Rep=Rep+doStuff_Rep(Models,REP_criterion,Prop_batch)
	end


	show_figure(list_out1, Log_Folder..'state'..name..'.log', 1000)

	return Temp/nb_sample,Prop/nb_sample, Rep/nb_sample
end

---------------------------------------------------------------------------------------
-- Function : Print_Loss(Temp_Train,Prop_Train,Rep_Train,Temp_Test,Prop_Test,Rep_Test,Log_Folder)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function Print_Loss(Temp_Train,Prop_Train,Rep_Train,Temp_Test,Prop_Test,Rep_Test,Log_Folder)
	show_loss(Temp_Train,Temp_Test, Log_Folder..'Temp_loss.log', 1000)
	show_loss(Prop_Train,Prop_Test, Log_Folder..'Prop_loss.log', 1000)
	show_loss(Rep_Train,Rep_Test, Log_Folder..'Rep_loss.log', 1000)
end

---------------------------------------------------------------------------------------
-- Function : load_list(list,lenght,height)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function load_list(list,lenght,height)
	im={}
	lenght=lenght or 200
	height=height or 200
	for i=1, #list do
		table.insert(im,getImage(list[i],lenght,height, false))
	end 
	return im
end

---------------------------------------------------------------------------------------
-- Function : getImage(im,length,height,SpacialNormalization)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function getImage(im,length,height,SpacialNormalization)
	if im=='' or im==nil then return nil end
	local image1=image.load(im,3,'byte')
	local format=length.."x"..height
	local img1_rsz=image.scale(image1,format)
	return preprocessing(img1_rsz,length,height, SpacialNormalization)
end

---------------------------------------------------------------------------------------
-- Function : show_loss(list_train, list_test, Name , scale)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function show_loss(list_train, list_test, Name , scale)

	local scale=scale or 1000
	-- log results to files
	accLogger = optim.Logger(Name)

	for i=1, #list_train do
	-- update logger
		accLogger:add{['train*'..scale] = list_train[i]*scale,['test*'..scale] = list_test[i]*scale}
	end
	-- plot logger
	accLogger:style{['train*'..scale] = '+',['test*'..scale] = '+'}
	accLogger:plot()
end
---------------------------------------------------------------------------------------
-- Function : show_figure(list_out1, Name , scale)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
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
