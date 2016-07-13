
function save_model(model,path)
	print("Saved at : "..path)
	model:cuda()
	parameters, gradParameters = model:getParameters()
	local lightModel = model:clone('weight','bias','running_mean','running_std'):double()
	torch.save(path,model)
end

function preprocessing(im)
		-- Name channels for convenience
	local channels = {'y','u','v'}
	local mean = {}
	local std = {}
	data = torch.Tensor( 3, 200, 200)
	  data:copy(im)
	--image.display{image=batch.data, legend='Avant'}
	for i,channel in ipairs(channels) do
	   -- normalize each channel globally:
	   mean[i] = data[i]:mean()
	   std[i] = data[{i,{},{}}]:std()
	   data[{i,{},{}}]:add(-mean[i])
	   data[{i,{},{}}]:div(std[i])
	end
	--image.display{image=batch.data, legend='Après'}

	--preprocessing data: normalize all three channels locally----------------

	-- Define the normalization neighborhood:
	local neighborhood = image.gaussian1D(5) -- 5 for face detector training

	-- Define our local normalization operator
	local normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1e-4)

	-- Normalize all channels locally:
	for c in ipairs(channels) do
	      data[{{c},{},{} }] = normalization:forward(data[{{c},{},{} }])
	end
	return data
end

function getBatch(list, indice, lenght, width, height, Type)
	
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
		Batch[1][i]=getImage(list.im1[start+i])
		Batch[2][i]=getImage(list.im2[start+i])
		if Type=="Prop" then
			Batch[3][i]=getImage(list.im3[start+i])
			Batch[4][i]=getImage(list.im4[start+i])
		end
	end

	return Batch

end

function Print_performance(Model,list1, epoch)
		local list_out1={}
		for i=1, #list1 do
			image1=getImage(list1[i])

			local Data1=image1:cuda()

			Model:forward(Data1)
			local State1=Model.output[1]		
				
			table.insert(list_out1,State1*1000)
		end
		show_figure(list_out1, epoch)
end

function getImage(im)
	if im=='' or im==nil then return nil end
	local image1=image.load(im,3,'byte')
	local img1_rsz=image.scale(image1,"200x200")
	return preprocessing(img1_rsz)
end

function show_figure(list_out1, epoch)
	-- log results to files
	accLogger = optim.Logger('./Log/state'..epoch..'.log')

	for i=1, #list_out1 do
	-- update logger
		accLogger:add{['out1'] = list_out1[i]}
	end
	-- plot logger
	accLogger:style{['Representation*1000'] = '-'}
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
