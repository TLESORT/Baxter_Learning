
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
	table.sort(list)
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

function Get_HeadCamera_HeadMvt()
	local Path="/home/lesort/baxter/original_data/"

	local Paths=Get_Folders(Path,'2016' .. '$')
	list={}
	list_txt={}
	for i=1, #Paths do
		list=Get_Folders(Paths[i],'head','bag' .. '$',list)
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
            error("data dimension for " .. rawCounter .. "th sample is not consistent with previous samples'")
        end
    
        if comment then labels = raw else table.insert(data,raw) end 
    end
    print(labels)
    return torch.Tensor(data), labels
end



-- A modifier on a rassemblé les images par joint et pas par delta....

function create_Training_list(list_im, txt)
	 local associated_images={im1={},im2={},im3={},im4={},Mode={}}

	tensor, label=tensorFromTxt(txt)

	for i=1, (#tensor[{}])[1] do
		-- arrondit au 1/10 près
		floor=math.floor(tensor[i][3]*10)/10
		ceil=math.ceil(tensor[i][3]*10)/10
		if math.abs(tensor[i][3]-ceil)>math.abs(tensor[i][3]-floor) then tensor[i][3]= floor
		else tensor[i][3]= ceil end
		print(tensor[i][3])
	end

	for i=1, #list_im-1 do
		value=tensor[i][3]
		for j=i+1, #list_im do
			if value==tensor[j][3] then
				table.insert(associated_images.im1,list_im[i])
				table.insert(associated_images.im2,list_im[j])
				table.insert(associated_images.im3,'')
				table.insert(associated_images.im4,'')
				table.insert(associated_images.Mode,'Prop')
			elseif value==(-1)*tensor[j][3] then
				table.insert(associated_images.im1,list_im[j])
				table.insert(associated_images.im2,list_im[i])
				table.insert(associated_images.im3,'')
				table.insert(associated_images.im4,'')
				table.insert(associated_images.Mode,'Prop')
			end
		
	
		end
	
	end

	for i=1, #associated_images.Mode do
		print('im1 : '..associated_images.im1[i])
		print('im2 : '..associated_images.im2[i])
		print('Mode : '..associated_images.Mode[i])
	end

end
