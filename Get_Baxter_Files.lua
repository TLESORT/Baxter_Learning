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
function Get_Folders(Path, including)
   local list= {}
   local list_txt={}
   for file in paths.files(Path) do
      if file:find(including) then
         Path_Folder= paths.concat(Path,file)
         table.insert(list,paths.concat(Path_Folder,"Images"))
         table.insert(list_txt, paths.concat(Path_Folder,"robot_joint_states.txt"))
      end
   end
   return list, list_txt
end


---------------------------------------------------------------------------------------
-- Function : Get_HeadCamera_HeadMvt(use_simulate_images)
-- Input (use_simulate_images) : boolean variable which say if we use or not simulate images 
-- Output (list_head_left): list of the images directories path
-- Output (list_txt):  txt list associated to each directories (this txt file contains the grundtruth of the robot position)
---------------------------------------------------------------------------------------
function Get_HeadCamera_HeadMvt()
   local Path="./Data/"
   local Paths_Folder, list_txt=Get_Folders(Path,'head_pan')

   table.sort(list_txt)
   table.sort(Paths_Folder)
   
   return Paths_Folder, list_txt
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
   local head_pan_indice=2
   tensor, label=tensorFromTxt(txt)
   local size=tensor:size(1)

   while WatchDog<100 do
      indice1=torch.random(1,size-1)
      indice2=indice1+1
      State1=tensor[indice1][head_pan_indice]
      State2=tensor[indice2][head_pan_indice]
      delta=State2-State1

      vector=torch.randperm(size) -- like this we sample uniformly the different possibility

      for i=1, size-1 do
         id=vector[i]
         State3=tensor[id][head_pan_indice]
         id2=id+1
         delta2=tensor[id2][head_pan_indice]-State3
         if not ((indice1==id and indice2==id2) or (indice1==id2 and indice2==id)) then
            if arrondit(delta2-delta)==0 then
               return {im1=indice1,im2=indice2,im3=id,im4=id2}
            elseif arrondit(delta2+delta)==0 then
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
function get_two_Prop_Pair(txt1, txt2)

   assert(txt1~=txt2) -- if you need prop image association from only one txt file see "get_one_random_Temp_Set(list_lenght)"
   local WatchDog=0
   local head_pan_indice=2

   -- tensor is a numImage x 20 features (head position, hand, arm angle etc ...)
   local tensor, label=tensorFromTxt(txt1)
   local tensor2, label=tensorFromTxt(txt2)

   -- print("txt2",txt2)
   -- io.read()

   -- print("tensor",tensor)
   -- print("tensor2",tensor2)
   -- io.read()

   local delta_action=0

   local size1=tensor:size(1)
   local size2=tensor2:size(1)

   vector=torch.randperm(size2-1)

   while WatchDog<100 do
      repeat
         indice1=torch.random(1,size1-1)
         indice2=indice1+1	
         State1=tensor[indice1][head_pan_indice]
         State2=tensor[indice2][head_pan_indice]
         delta=State2-State1
      until(delta~=0)

      for i=1, size2-1 do
         id=vector[i]
         State3=tensor2[id][head_pan_indice]
         id2=id+1
         State4=tensor2[id2][head_pan_indice]
         delta2=State4-State3
         if arrondit(delta2-delta)==0 then
            return {im1=indice1,im2=indice2,im3=id,im4=id2}
         elseif arrondit(delta2+delta)==0 then
            delta_action=(delta2-delta)^2
            return {im1=indice1,im2=indice2,im3=id2,im4=id}
         end
      end
      WatchDog=WatchDog+1
   end
   print("PROP WATCHDOG ATTACK!!!!!!!!!!!!!!!!!!")
   assert(false, "Get_Baxter_File.lua line 207, database is broken")
end



local function isRewardJoint(joint, ListRewardJoint)
   local isReward=false
   for i=1, #ListRewardJoint do
      if arrondit(joint)==ListRewardJoint[i] then
         isReward=true
         break
      end
   end
   return isReward
end
-- I need to search images representing a starting state.
-- then the same action applied to this to state (the same variation of joint) should lead to a different reward.
-- for instance we choose for reward the fact to have a joint = 0

-- NB : the two states will be took in different list but the two list can be the same

function get_one_random_Caus_Set(txt1, txt2,use_simulate_images)
   local WatchDog=0
   local head_pan_indice=2
   local tensor, label=tensorFromTxt(txt1)
   local tensor2, label=tensorFromTxt(txt2)
   local rewarded_Joint = {}
   
   if string.find(txt1, "moreData") then
      rewarded_Joint={-1.3,1.3,-1.4,1.4}
   else
      rewarded_Joint={-0.8,0.8}
   end

   local size1=tensor:size(1)
   local size2=tensor2:size(1)

   while WatchDog<500 do
      repeat
         indice1=torch.random(1,size1-1)			
         State1=tensor[indice1][head_pan_indice]
         indice2=indice1+1
         State2=tensor[indice2][head_pan_indice]
      until(not(isRewardJoint(State1,rewarded_Joint)or isRewardJoint(State2,rewarded_Joint)))

      delta=State2-State1

      vector=torch.randperm(size2-1)

      for i=1, size2-1 do
         id=vector[i]
         State3=tensor2[id][head_pan_indice]
         id2=id+1
         State4=tensor2[id2][head_pan_indice]
         delta2=State4-State3
         if arrondit(delta2-delta)==0 and 
            isRewardJoint(State4,rewarded_Joint)and
         not(isRewardJoint(State3,rewarded_Joint)) then
            return {im1=indice1,im2=id}
         elseif arrondit(delta2+delta)==0 and
            isRewardJoint(State3,rewarded_Joint)and
         not(isRewardJoint(State4,rewarded_Joint)) then		
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
function getTruth(txt)
   local truth={}
   local head_pan_indice=2
   local tensor, label=tensorFromTxt(txt)
   
   for i=1, (#tensor[{}])[1] do
      table.insert(truth, tensor[i][head_pan_indice])
   end
   return truth
end


---------------------------------------------------------------------------------------
-- Function : arrondit(value)
-- Input (tensor) : 
-- Input (head_pan_indice) : 
-- Output (tensor): 
---------------------------------------------------------------------------------------
function arrondit(value)
   floor=math.floor(value*10)/10
   ceil=math.ceil(value*10)/10
   if math.abs(value-ceil)>math.abs(value-floor) then result=floor
   else result=ceil end
   return result
end
