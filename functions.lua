
function save_model(model,path)
	print("Saved at : "..path)
	model:cuda()
	parameters, gradParameters = model:getParameters()
	local lightModel = model:clone():float()
	lightModel:clearState()
	torch.save(path,lightModel)
end
function load_list(list, train)
	local im={}
	local lenght=image_width or 200
	local height=image_height or 200
	for i=1, #list do
		table.insert(im,getImage(list[i],lenght,height,train))
	end 
	return im
end

function getImage(im,length,height, train)
	if im=='' or im==nil then return nil end
	local image1=image.load(im,3,'byte')
	local format=length.."x"..height
	local img1_rsz=image.scale(image1,format)
	return preprocessing(img1_rsz,length,height, train)
end
function preprocessing(im, lenght, width,train)

		-- Name channels for convenience
	local channels = {'r','g','b'}
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
	if train then data=dataAugmentation(data, lenght, width) end

	return data
end

local function gamma(im)
	local Gamma= torch.Tensor(3,3)
	local channels = {'r','g','b'}
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

local function transformation(im, v,e)
	local transfo=torch.Tensor(3,200,200)
	local Gamma=torch.mv(v,e)
	for i=1, 3 do
		transfo[i]=im[i]+Gamma[i]
	end
 return transfo
end

function dataAugmentation(im, lenght, width)
	local channels = {'r','g','b'}

	gam=gamma(im)
	e, V = torch.eig(gam,'V')
	factors=torch.randn(3)*0.1
	for i=1,3 do e:select(2, 1)[i]=e:select(2, 1)[i]*factors[i] end
	im=transformation(im, V,e:select(2, 1))
	noise=torch.rand(3,lenght,width)
	local mean = {}
	local std = {}
	for i,channel in ipairs(channels) do
	   -- normalize each channel globally:
	   mean[i] = noise[i]:mean()
	   std[i] = noise[{i,{},{}}]:std()
	   noise[{i,{},{}}]:add(-mean[i])
	   noise[{i,{},{}}]:div(std[i])
	end
	return im+noise
end


function getRandomBatch(imgs1, imgs2, txt1, txt2, lenght, Mode)

	local width=image_width or 200
	local height=image_height or 200

	if Mode=="Prop" or Mode=="Rep" then
		Batch=torch.Tensor(4, lenght,3, width, height)
	else
		Batch=torch.Tensor(2, lenght,3, width, height)
	end
	
	for i=1, lenght do
		if Mode=="Prop" or Mode=="Rep" then
			if txt1==txt2 then Set=get_one_random_Prop_Set(txt1)
			else Set=get_two_Prop_Pair(txt1, txt2) end
			Batch[1][i]=imgs1[Set.im1]
			Batch[2][i]=imgs1[Set.im2]
			Batch[3][i]=imgs2[Set.im3]
			Batch[4][i]=imgs2[Set.im4]
		elseif Mode=="Temp" then
			Set=get_one_random_Temp_Set(#imgs1)
			Batch[1][i]=imgs1[Set.im1]
			Batch[2][i]=imgs1[Set.im2]
		elseif Mode=="Caus" then
			Set=get_one_random_Caus_Set(txt1, txt2)
			Batch[1][i]=imgs1[Set.im1]
			Batch[2][i]=imgs2[Set.im2]
		else
			print "getRandomBatch Wrong mode "
		end
	end
	return Batch

end


