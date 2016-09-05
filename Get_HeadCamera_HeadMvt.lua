
---------------------------------------------------------------------------------------
-- Function : images_Paths(path)
-- Input (Path): path of a Folder which contained jpg images
-- Output : list of the jpg files path
---------------------------------------------------------------------------------------
function images_Paths(Path)
	local listImage={}
	for file in paths.files(Path) do
	   -- We only load files that match the extension
	   if file:find('jpg' .. '$') then
	      -- and insert the ones we care about in our table
	      table.insert(listImage, paths.concat(Path,file))
	   end
	   
	end
	table.sort(listImage)
	return listImage
end


---------------------------------------------------------------------------------------
-- Function : 
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function txt_path(Path)
	local txt=nil
	for file in paths.files(Path) do
	   if file:find('txt' .. '$') then
	      txt=paths.concat(Path,file)
	   end
	end
	return txt
end

---------------------------------------------------------------------------------------
-- Function : 
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function Get_Folders(Path, including, excluding,list)
	local list=list or {}
	local incl=including or ""
	local excl=excluding or "uyfouhjbhytfoughl" -- random motif
	for file in paths.files(Path) do
	   -- We only load files that match 2016 because we know that there are the folder we are interested in
	   if file:find(incl) and (not file:find(excl)) then
	      -- and insert the ones we care about in our table
	      table.insert(list, paths.concat(Path,file))
	   end
	end
	return list
end


---------------------------------------------------------------------------------------
-- Function : Get_HeadCamera_HeadMvt(use_simulate_images)
-- Input (use_simulate_images) : boolean variable which say if we use or not simulate images 
-- Output (list_head_left): list of the images directories path
-- Output (list_txt):  txt list associated to each directories (this txt file contains the grundtruth of the robot position)
---------------------------------------------------------------------------------------
function Get_HeadCamera_HeadMvt(use_simulate_images)
	local use_simulate_images=use_simulate_images or false
	local Path
	if use_simulate_images then
		Path=paths.home.."/baxter_sim_data/original_data/"
	else
		 Path=paths.home.."/baxter/original_data/"
	end

	local Paths=Get_Folders(Path,'2016$')
	list={}
	list_txt={}
	for i=1, #Paths do
		list=Get_Folders(Paths[i],'head','bag$',list)
	end

	list_head_left={}
	for i=1, #list do
		list_head_left=Get_Folders(list[i],'cameras_head',nil,list_head_left)
		table.insert(list_txt, txt_path(list[i]))
	end
	table.sort(list_txt)
	table.sort(list_head_left)
	return list_head_left, list_txt
end


---------------------------------------------------------------------------------------
-- Function : tensorFromTxt(path)
-- Input (path) : path of a txt file which contain position of the robot
-- Output (torch.Tensor(data)): tensor with all the joint values (col: joint, lign : indice)
-- Output (labels):  name of the joint
---------------------------------------------------------------------------------------
function tensorFromTxt(path)
    local data, raw = {}, {}
    local rawCounter, columnCounter = 0, 0
    local nbFields, labels, _line = nil, nil, nil

    for line in io.lines(path)  do 
        local comment = false
        if line:sub(1,1)=='#' then  
            comment = true            
            line = line:sub(2)
        end 
        rawCounter = rawCounter +1      
        columnCounter=0
        raw = {}
        for value in line:gmatch'%S+' do
            columnCounter = columnCounter+1
            raw[columnCounter] = tonumber(value)
        end

        -- we check that every row contains the same number of data
        if rawCounter==1 then
            nbFields = columnCounter
        elseif columnCounter ~= nbFields then
            error("data dimension for " .. rawCounter .. "the sample is not consistent with previous samples'")
        end
    
        if comment then labels = raw else table.insert(data,raw) end 
    end
    return torch.Tensor(data), labels
end


---------------------------------------------------------------------------------------
-- Function : get_one_random_Temp_Set(list_im)
-- Input (list_lenght) : lenght of the list of images
-- Output : 2 indices of images which are neightboor in the list (and in time) 
---------------------------------------------------------------------------------------
function get_one_random_Temp_Set(list_lenght)
	indice=torch.random(1,list_lenght-1)
	return {im1=indice,im2=indice+1}
end

function get_one_random_Prop_Set(txt,use_simulate_images)
	local WatchDog=0
	local head_pan_indice=3
	if use_simulate_images then head_pan_indice=2 end
	tensor, label=tensorFromTxt(txt)
	tensor=tensor_arrondit(tensor, head_pan_indice)
	local size=tensor:size(1)


	local ecart=torch.random(1,2)

	while WatchDog<100 do
		indice1=torch.random(1,size-ecart)
		indice2=indice1+ecart
		State1=tensor[indice1][head_pan_indice]
		State2=tensor[indice2][head_pan_indice]
		delta=State1-State2

		vector=torch.randperm(size-ecart) -- like this we sample uniformly the different possibility

		for i=1, size-ecart do
			id=vector[i]
			State3=tensor[id][head_pan_indice]
			id2=id+ecart
			delta2=State3-tensor[id2][head_pan_indice]
			if not ((indice1==id and indice2==id2)or (indice1==id2 and indice2==id)) then
				if delta2==delta then
					return {im1=indice1,im2=indice2,im3=id,im4=id2}
				elseif delta2==delta*-1 then
					return {im1=indice1,im2=indice2,im3=id2,im4=id}
				end
			end
		end
		WatchDog=WatchDog+1
	end
end

---------------------------------------------------------------------------------------
-- Function : get_two_Prop_Pair(txt1, txt2,use_simulate_images)
-- Input (txt1) : path of the file of the first list of joint
-- Input (txt2) : path of the file of the second list of joint
-- Input (use_simulate_images) : boolean variable which say if we use or not simulate images (we need this information because the data is not formated exactly the same in the txt file depending on the origin of images)
-- Output : structure with 4 indices which represente a quadruplet (2 Pair of images from 2 different list) for Traininng with prop prior. The variation of joint for on pair should be the same as the variation for the second
---------------------------------------------------------------------------------------
function get_two_Prop_Pair(txt1, txt2,use_simulate_images)

	assert(txt1~=txt2) -- if you need prop image association from only one txt file see "get_one_random_Temp_Set(list_lenght)"
	local WatchDog=0
	local head_pan_indice=3
	if use_simulate_images then head_pan_indice=2 end
	local tensor, label=tensorFromTxt(txt1)
	local tensor2, label=tensorFromTxt(txt2)

	local ecart=torch.random(1,2)

	local delta_action=0

	local size1=tensor:size(1)
	local size2=tensor2:size(1)


	vector=torch.randperm(size2-ecart)

	while WatchDog<100 do
		indice1=torch.random(1,size1-ecart)
		indice2=indice1+ecart	
		State1=tensor[indice1][head_pan_indice]
		State2=tensor[indice2][head_pan_indice]
		delta=State2-State1

		for i=1, size2-ecart do
			id=vector[i]
			State3=tensor2[id][head_pan_indice]
			id2=id+ecart
			State4=tensor2[id2][head_pan_indice]
			delta2=State4-State3
			if arrondit(delta2-delta)==0 then
				delta_action=(delta2-delta)^2
				return {im1=indice1,im2=indice2,im3=id,im4=id2, delta=delta_action}
			elseif arrondit(delta2+delta)==0 then
				delta_action=(delta2-delta)^2
				return {im1=indice1,im2=indice2,im3=id2,im4=id, delta=delta_action}
			end
		end
		WatchDog=WatchDog+1
	end
	print("PROP WATCHDOG ATTACK!!!!!!!!!!!!!!!!!!")
end



-- I need to search images representing a starting state.
-- then the same action applied to this to state (the same variation of joint) should lead to a different reward.
-- for instance we choose for reward the fact to have a joint = 0

-- NB : the two states will be took in different list but the two list can be the same

function get_one_random_Caus_Set(txt1, txt2,use_simulate_images)
	local WatchDog=0
	local head_pan_indice=3
	if use_simulate_images then head_pan_indice=2 end
	local tensor, label=tensorFromTxt(txt1)
	local tensor2, label=tensorFromTxt(txt2)

	local rewarded_Joint=0.5
	local rewarded_Joint2=-0.5
	local rewarded_Joint3=0.5

	local ecart=torch.random(1,2)

	local size1=tensor:size(1)
	local size2=tensor2:size(1)

	while WatchDog<200 do
		repeat
			indice1=torch.random(1,size1-ecart)			
			State1=tensor[indice1][head_pan_indice]
			indice2=indice1+ecart
			State2=tensor[indice2][head_pan_indice]
		until(arrondit(State1)~=rewarded_Joint
			and arrondit(State1)~=rewarded_Joint2
			and arrondit(State1)~=rewarded_Joint3
			and arrondit(State2)~=rewarded_Joint
			and arrondit(State2)~=rewarded_Joint2
			and arrondit(State2)~=rewarded_Joint3)

		delta=State2-State1

		vector=torch.randperm(size2-ecart)

		for i=1, size2-ecart do
			id=vector[i]
			State3=tensor2[id][head_pan_indice]
			id2=id+ecart
			State4=tensor2[id2][head_pan_indice]
			delta2=State4-State3
			if arrondit(delta2-delta)==0 and 
				(arrondit(State4)==rewarded_Joint 
				or arrondit(State4)==rewarded_Joint2 
				or arrondit(State4)==rewarded_Joint3) then
					return {im1=indice1,im2=id}
			elseif arrondit(delta2+delta)==0 and
				(arrondit(State3)==rewarded_Joint 
				or arrondit(State3)==rewarded_Joint2 
				or arrondit(State3)==rewarded_Joint3) then		
					return {im1=indice1,im2=id2}
			end
		end
		WatchDog=WatchDog+1
	end
	print("CAUS WATCHDOG ATTACK!!!!!!!!!!!!!!!!!!")
end

---------------------------------------------------------------------------------------
-- Function : getTruth(txt,use_simulate_images)
-- Input (txt) : 
-- Input (use_simulate_images) : 
-- Input (arrondit) :
-- Output (truth): 
---------------------------------------------------------------------------------------
function getTruth(txt,use_simulate_images, Arrondit)
	local truth={}
	local Arrondit= Arrondit or false
	local head_pan_indice=3
	if use_simulate_images then head_pan_indice=2 end
	local tensor, label=tensorFromTxt(txt)
	if Arrondit then tensor=tensor_arrondit(tensor,head_pan_indice) end
	
	for i=1, (#tensor[{}])[1] do
		table.insert(truth, tensor[i][head_pan_indice])
	end
	return truth
end


---------------------------------------------------------------------------------------
-- Function : arrondit(tensor, head_pan_indice)
-- Input (tensor) : 
-- Input (head_pan_indice) : 
-- Output (tensor): 
---------------------------------------------------------------------------------------
function tensor_arrondit(tensor, head_pan_indice)
	for i=1, (#tensor[{}])[1] do
		tensor[i][head_pan_indice]=arrondit(tensor[i][head_pan_indice])
	end
	return tensor
end

function arrondit(value)
	floor=math.floor(value*10)/10
	ceil=math.ceil(value*10)/10
	if math.abs(value-ceil)>math.abs(value-floor) then result=floor
	else result=ceil end
	return result
end
