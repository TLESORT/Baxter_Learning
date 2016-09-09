---------------------------------------------------------------------------------------
-- Function :save_model(model,path) 
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
-- Function : preprocessing(im, lenght, width, SpacialNormalization)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function preprocessing(im, lenght, width,train)

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
	--if train then data=dataAugmentation(data, lenght, width) end-- change here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
-------------------------------------------------------------------------------------------------------
data2 = torch.Tensor( 1, lenght, width)-- change here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
data2[1]=data[1]
-------------------------------------------------------------------------------------------------------
	return data2
end

local function gamma(im)
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

local function transformation(im, v,e)
	local transfo=torch.Tensor(3,200,200)
	local Gamma=torch.mv(v,e)
	for i=1, 3 do
		transfo[i]=im[i]+Gamma[i]
	end
 return transfo
end

---------------------------------------------------------------------------------------
-- Function : dataAugmentation(im, lenght, width)
-- Input ():
-- Output ():
-- goal : By using data augmentation we want or network to be more resistant to no task relevant perturbations like luminosity variation or noise
---------------------------------------------------------------------------------------
function dataAugmentation(im, lenght, width)
	local channels = {'y','u','v'}

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
	return im--+noise
end

---------------------------------------------------------------------------------------
-- Function :getBatch(imgs, list, indice, lenght, width, height, Type) 
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
		Batch=torch.Tensor(4, lenght,1, width, height)
	else
		Batch=torch.Tensor(2, lenght,1, width, height)
	end
	
	for i=1, lenght do
		Batch[1][i]=imgs[list.im1[start+i][1]]
		Batch[2][i]=imgs[list.im2[start+i][1]]
		if Type=="Prop" then
			Batch[3][i]=imgs[list.im3[start+i][1]]
			Batch[4][i]=imgs[list.im4[start+i][1]]
		end
	end

	return Batch

end

---------------------------------------------------------------------------------------
-- Function : representation_error(list_real, list_estimation)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function representation_error(list_real, list_estimation)

	Somme_Real=0
	Somme_Estimation=0

	assert(#list_real==#list_estimation)	
	
	--process cste normalisation
	for i=1, #list_real do
		Somme_Real=Somme_Real+list_real[i]
		Somme_Estimation=Somme_Estimation+list_estimation[i]
	end

	Moyenne_Real=Somme_Real/#list_real
	Moyenne_Estimation=Somme_Estimation/#list_estimation
	erreur=0
	erreur_tot=0
	for j=1, #list_real do
		State=(list_real[j]-Moyenne_Real)/Somme_Real
		State_Estimation=(list_estimation[j]-Moyenne_Estimation)/Somme_Estimation
		erreur=math.abs((State)-(State_Estimation)/State)
		erreur_tot=erreur_tot+erreur
	end

	return (erreur_tot/#list_real)*100
end

---------------------------------------------------------------------------------------
-- Function : getRandomBatch(imgs, txt, lenght, width, height, Mode, use_simulate_images)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function getRandomBatch(imgs, txt, lenght, width, height, Mode, use_simulate_images)
	
	if Mode=="Prop" or Mode=="Rep" then
		Batch=torch.Tensor(4, lenght,1, width, height)-- change here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	else
		Batch=torch.Tensor(2, lenght,1, width, height)-- change here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	end
	
	for i=1, lenght do
		if Mode=="Prop" or Mode=="Rep" then
			Set=get_one_random_Prop_Set(txt ,use_simulate_images)
			Batch[1][i][1]=imgs[Set.im1][1]-- change here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			Batch[2][i][1]=imgs[Set.im2][1]-- change here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			Batch[3][i][1]=imgs[Set.im3][1]-- change here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			Batch[4][i][1]=imgs[Set.im4][1]-- change here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		elseif Mode=="Temp" then
			Set=get_one_random_Temp_Set(#imgs)
			Batch[1][i][1]=imgs[Set.im1][1]-- change here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			Batch[2][i][1]=imgs[Set.im2][1]-- change here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		elseif Mode=="Caus" then
			Set=get_one_random_Caus_Set(txt, txt, use_simulate_images)
			Batch[1][i][1]=imgs[Set.im1][1]-- change here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			Batch[2][i][1]=imgs[Set.im2][1]-- change here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		else
			print "getRandomBatch Wrong mode "
		end
	end
	return Batch
end


---------------------------------------------------------------------------------------
-- Function :	Have_Todo(list_prior,prior)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function Have_Todo(list_prior,prior)
	local answer=false
	if #list_prior~=0 then
		for i=1, #list_prior do
			if list_prior[i]==prior then answer=true end
		end
	end
	return answer
end


---------------------------------------------------------------------------------------
-- Function :	Get_Folder_Name(Log_Folder,Prior_Used)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function Get_Folder_Name(Log_Folder,list_prior)
	name=''
	if #list_prior~=0 then
		if #list_prior==1 then
			name=list_prior[1].."_Only"
		elseif #list_prior==4 then
			name='Everything'
		else
			name=list_prior[1]
			for i=2, #list_prior do
				name=name..'_'..list_prior[i]
			end
		end
	end
	return Log_Folder..name..'/'
end

---------------------------------------------------------------------------------------
-- Function :getRandomBatchFromSeparateList(imgs1, imgs2, txt1, txt2, lenght, image_width, image_height, Mode, use_simulate_images) 
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function getRandomBatchFromSeparateList(imgs1, imgs2, txt1, txt2, lenght, image_width, image_height, Mode, use_simulate_images)

	local width=image_width or 200
	local height=image_height or 200

	if Mode=="Prop" or Mode=="Rep" then
		Batch=torch.Tensor(4, lenght,1, width, height)-- change here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	else
		Batch=torch.Tensor(2, lenght,1, width, height)-- change here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	end
	
	for i=1, lenght do
		if Mode=="Prop" or Mode=="Rep" then
			Set=get_two_Prop_Pair(txt1, txt2, use_simulate_images)
			Batch[1][i][1]=imgs1[Set.im1][1]-- change here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			Batch[2][i][1]=imgs1[Set.im2][1]-- change here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			Batch[3][i][1]=imgs2[Set.im3][1]-- change here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			Batch[4][i][1]=imgs2[Set.im4][1]-- change here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		elseif Mode=="Temp" then
			Set=get_one_random_Temp_Set(#imgs1)
			Batch[1][i][1]=imgs1[Set.im1][1]-- change here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			Batch[2][i][1]=imgs1[Set.im2][1]-- change here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		elseif Mode=="Caus" then
			Set=get_one_random_Caus_Set(txt1, txt2, use_simulate_images)
			Batch[1][i][1]=imgs1[Set.im1][1]-- change here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			Batch[2][i][1]=imgs2[Set.im2][1]-- change here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		else
			print "getRandomBatchFromSeparateList Wrong mode "
		end
	end
	return Batch

end

---------------------------------------------------------------------------------------
-- Function :copy_weight(model, AE)
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
function real_loss(txt,use_simulate_images)

	local REP_criterion=get_Rep_criterion()
	local PROP_criterion=get_Prop_criterion()
	local CAUS_criterion=get_Caus_criterion()
	local TEMP_criterion=nn.MSDCriterion()
	
	local truth=getTruth(txt,use_simulate_images)

	local temp_loss=0
	local prop_loss=0
	local rep_loss=0
	local caus_loss=0

	local nb_sample=100

	for i=0, nb_sample do
		Set_prop=get_one_random_Prop_Set(txt ,use_simulate_images)
		Set_temp=get_one_random_Temp_Set(#truth)
		Caus_temp=get_one_random_Caus_Set(txt, txt, use_simulate_images)

		joint1=torch.Tensor(1)
		joint2=torch.Tensor(1)
		joint3=torch.Tensor(1)
		joint4=torch.Tensor(1)

		joint1[1]=truth[Caus_temp.im1]
		joint2[1]=truth[Caus_temp.im2]		
		caus_loss=caus_loss+CAUS_criterion:updateOutput({joint1, joint2})

		joint1[1]=truth[Set_temp.im1]
		joint2[1]=truth[Set_temp.im2]		
		temp_loss=temp_loss+TEMP_criterion:updateOutput({joint1, joint2})

--print("TEMP: "..TEMP_criterion:updateOutput({joint1, joint2}))
		joint1[1]=truth[Set_prop.im1]
		joint2[1]=truth[Set_prop.im2]
		joint3[1]=truth[Set_prop.im3]
		joint4[1]=truth[Set_prop.im4]
		prop_loss=prop_loss+PROP_criterion:updateOutput({joint1, joint2, joint3, joint4})
		rep_loss=rep_loss+REP_criterion:updateOutput({joint1, joint2, joint3, joint4})
		
		--print("PROP	: "..PROP_criterion:updateOutput({joint1, joint2, joint3, joint4})[1])
		--print("REP		: "..REP_criterion:updateOutput({joint1, joint2, joint3, joint4})[1])
--print("-----------------------------------------")
	end

	return temp_loss/nb_sample, prop_loss/nb_sample, rep_loss/nb_sample, caus_loss/nb_sample
end



---------------------------------------------------------------------------------------
-- Function : load_list(list,lenght,height)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function load_list(list,lenght,height, train)
	im={}
	lenght=lenght or 200
	height=height or 200
	for i=1, #list do
		table.insert(im,getImage(list[i],lenght,height,train))
	end 
	return im
end

---------------------------------------------------------------------------------------
-- Function : getImage(im,length,height,SpacialNormalization)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function getImage(im,length,height, train)
	if im=='' or im==nil then return nil end
	local image1=image.load(im,3,'byte')
	local format=length.."x"..height
	local img1_rsz=image.scale(image1,format)
	return preprocessing(img1_rsz,length,height, train)
end


