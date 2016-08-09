
---------------------------------------------------------------------------------------
-- Function : images_Paths(path)
-- Input : path of a Folder which contained jpg images
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



function txt_path(Path)
	local txt=nil
	for file in paths.files(Path) do
	   if file:find('txt' .. '$') then
	      txt=paths.concat(Path,file)
	   end
	end
	return txt
end


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


function get_one_random_SetOfImages(list_im, txt,use_simulate_images, Mode)
	if Mode=="Temp" then
		 Set=get_one_random_Temp_Set(list_im)
	elseif Mode=="Prop" or Mode=="Prep" then		
		 Set=get_one_random_Prop_Set(list_im, txt,use_simulate_images, Mode)
	elseif Mode=="Caus" then
		--TODO
	else print("Wrong mode used in get_one_random_SetOfImages(Mode)")
	end
end

function get_one_random_Temp_Set(list_im)
	indice=math.random(1,#list_im-1)
	return {im1=indice,im2=indice+1,im3=0,im4=0,Mode='Temp'}
end

function get_one_random_Prop_Set(list_im, txt,use_simulate_images,Mode)
	local WatchDog=0
	local head_pan_indice=3
	if use_simulate_images then head_pan_indice=2 end
	tensor, label=tensorFromTxt(txt)
	tensor=arrondit(tensor, head_pan_indice)

	while WatchDog<100 do
		indice1=math.random(1,#list_im-1)
		indice2=math.random(1,#list_im-1)
		State1=tensor[indice1][head_pan_indice]
		State2=tensor[indice2][head_pan_indice]
		delta=State1-State2

		vector=torch.randperm(#list_im) -- like this we sample uniformly the different possibility

		for i=1, #list_im-1 do
			id=vector[i]
			State3=tensor[id][head_pan_indice]
			for j=i+1, #list_im do
				id2=vector[j]
				delta2=State3-tensor[id2][head_pan_indice]
				if not ((indice1==id and indice2==id2)or (indice1==id2 and indice2==id)) then
					if delta2==delta then
						return {im1=indice1,im2=indice2,im3=id,im4=id2,Mode=Mode}
					elseif delta2==delta*-1 then
						return {im1=indice1,im2=indice2,im3=id2,im4=id,Mode=Mode}
					end
				end
			end
		end
		WatchDog=WatchDog+1
	end
end

function getTruth(txt,use_simulate_images)
	local truth={}
	local tensor, label=tensorFromTxt(txt)
	 local head_pan_indice=3
	 if use_simulate_images then head_pan_indice=2 end
	
	for i=1, (#tensor[{}])[1] do
		table.insert(truth, tensor[i][head_pan_indice])
	end

	return truth
end

function arrondit(tensor, head_pan_indice)
	for i=1, (#tensor[{}])[1] do
		floor=math.floor(tensor[i][head_pan_indice]*10)/10
		ceil=math.ceil(tensor[i][head_pan_indice]*10)/10
		if math.abs(tensor[i][head_pan_indice]-ceil)>math.abs(tensor[i][head_pan_indice]-floor) then 
			tensor[i][head_pan_indice]= floor
		else tensor[i][head_pan_indice]= ceil end
	end
	return tensor
end

-- create list of image indice which symboliseimages association for prior training
function create_Head_Training_list(list_im, txt,use_simulate_images)
	 local associated_images_Prop={im1={},im2={},im3={},im4={},Mode={}}
	 local associated_images_Temp={im1={},im2={},im3={},im4={},Mode={}}
	 local head_pan_indice=3
	 if use_simulate_images then head_pan_indice=2 end
	
	local truth=getTruth(txt,use_simulate_images)

	local tensor, label=tensorFromTxt(txt)
	tensor=arrondit(tensor,head_pan_indice)

-- TEMP : ici il est considéré que deux états proches sont potentiellement proche dans le temps
	for i=1, #list_im do

		--[[
		value=tensor[i][head_pan_indice]
		for j=i+1, #list_im do
			if value==tensor[j][head_pan_indice] then
				table.insert(associated_images_Temp.im1,i)
				table.insert(associated_images_Temp.im2,j)
				table.insert(associated_images_Temp.im3,0)
				table.insert(associated_images_Temp.im4,0)
				table.insert(associated_images_Temp.Mode,'Temp')
			end	
		end
		--]]

-- we add every two images which are temporaly correlated
-- it might add associated images that already linked before but it's not a problem
--because we don't have a lot of images for temporal coherence.
		if i<#list_im-1 then
			table.insert(associated_images_Temp.im1,i)
			table.insert(associated_images_Temp.im2,i+1)
			table.insert(associated_images_Temp.im3,0)
			table.insert(associated_images_Temp.im4,0)
			table.insert(associated_images_Temp.Mode,'Temp')

				--not the same for no-twins learning			
			table.insert(associated_images_Temp.im1,i+1)
			table.insert(associated_images_Temp.im2,i)
			table.insert(associated_images_Temp.im3,0)
			table.insert(associated_images_Temp.im4,0)
			table.insert(associated_images_Temp.Mode,'Temp')
		end
	
	end
 -- PROP
	for i=1, #list_im-1 do
		value=tensor[i][head_pan_indice]
		for j=i+1, #list_im do
			delta=value-tensor[j][head_pan_indice]
			for l=i, #list_im do
				value2=tensor[l][head_pan_indice]
				for m=l+1, #list_im do
					delta2=value2-tensor[m][head_pan_indice]
					if (l~=i or m~=j) and (delta==delta2) and (delta2*delta)~=0 then
						table.insert(associated_images_Prop.im1,i)
						table.insert(associated_images_Prop.im2,j)
						table.insert(associated_images_Prop.im3,l)
						table.insert(associated_images_Prop.im4,m)
						table.insert(associated_images_Prop.Mode,'Prop')

					--print(delta.."delta1")
					--print(delta2.."delta2")
	
					elseif delta==(-1)*delta2 and delta~=0 then
						table.insert(associated_images_Prop.im1,i)
						table.insert(associated_images_Prop.im2,j)
						table.insert(associated_images_Prop.im3,m)
						table.insert(associated_images_Prop.im4,l)
						table.insert(associated_images_Prop.Mode,'Prop')
					end

					
				end
			end
	
		end
	
	end
print("Nb Temp association : "..#associated_images_Temp.Mode)
print("Nb Prop association : "..#associated_images_Prop.Mode)



return associated_images_Prop, associated_images_Temp, truth
end
