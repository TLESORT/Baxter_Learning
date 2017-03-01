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
require 'functions'
require 'printing'
require "Get_Baxter_Files"
require 'priors'

require('./models/convolutionnal')

local function getRandomBatch(imgs,list_txt, sizeBatch)

   local numSeq = #list_txt

   local indice1=torch.random(1,numSeq-1)
   local txt=list_txt[indice1]
   local imgsTemp=imgs[indice1]


   local truth=getTruth(txt)
   
   local batchList = {}
   local yList = {}

   clh = imgsTemp[1]:size() -- Channel, Length, Height
   batch = torch.zeros(sizeBatch,clh[1],clh[2],clh[3])
   
   for i=1,sizeBatch do
      local id=torch.random(1,#imgsTemp)
      batch[i] = imgsTemp[id]
      yList[i] = truth[id] 
   end

   local y = torch.Tensor(yList)

   batch = batch:cuda()
   y = y:cuda()

   return batch, y

end

function Rico_Training(model,batch,y,LR)
   local LR=LR or 0.001
   local optimizer = optim.rmsprop
   -- local criterion = nn.MSECriterion():cuda()
   local criterion = nn.SmoothL1Criterion():cuda()
   
   -- create closure to evaluate f(X) and df/dX
   local feval = function(x)
      -- just in case:
      collectgarbage()
      -- get new parameters
      if x ~= parameters then
         parameters:copy(x)
      end
      -- reset gradients
      gradParameters:zero()

      
      local yhat = model:forward(batch)
      local loss = criterion:forward(yhat,y)

      -- if i>100 then
      --    print("x1",batch[1][{1,1,{1,5}}])

      --    print("x2",batch[2][{1,1,{1,5}}])

      --    print("yhat",yhat)
      --    print("y",y)
      --    io.read()
      -- end

      local grad = criterion:backward(yhat,y)
      model:backward(batch, grad)
      
      return loss,gradParameters
   end
   
   optimState={learningRate=LR}
   parameters, loss=optimizer(feval, parameters, optimState)
   return loss[1]
end

function accuracy(imgs_test,truth)
   local acc = 0

   local x = torch.Tensor(#imgs_test,imgs_test[1]:size(1),imgs_test[1]:size(2),imgs_test[1]:size(3))

   for i=1,#imgs_test do x[i]=imgs_test[i] end
   local yhat = model:forward(x:cuda())

   print("yhat",yhat[1][1],yhat[2][1],yhat[3][1],yhat[4][1],yhat[60][1],yhat[61][1])
   print("y",truth[1],truth[2],truth[3],truth[4],truth[60],truth[61])
   
   for i=1,#imgs_test do
      acc = acc + math.sqrt(math.pow(yhat[i][1]-truth[i],2))
   end
   return acc
end


function train_Epoch(model,list_folders_images,list_txt,Log_Folder,use_simulate_images,LR)

   local sizeBatch=40
   local nbEpoch=100
   local nbBatch=15
   local name_save=Log_Folder..'HeadSupervised.t7'

   --to normalize images, you need them, 'preprocessing' function calculate those values
   local mean = 0 
   local std = 0 

   nbList= #list_folders_images

   print("Loading Images")
   reconstruct = false
   imgs, std, mean = loadTrain(reconstruct)
   print("Training")

   -- we use last list as test 
   local list_truth=images_Paths(list_folders_images[nbList])
   local imgs_test=load_list(list_truth,image_width,image_height,false)

   local imgs_test = preprocessingTest(imgs_test,mean,std)

   local txt_test=list_txt[nbList]
   local truth=getTruth(txt_test,use_simulate_images)

   assert(#imgs_test==#truth,"Different number of images and corresponding ground truth, something is wrong")
   
   --for epoch=1, nbEpoch do
   for epoch=1,1 do

      print('--------------Epoch : '..epoch..' ---------------')
      local lossTemp=0

      for numBatch=1, nbBatch do
         Batch_Temp, y =getRandomBatch(imgs, list_txt, sizeBatch)
         lossTemp = lossTemp + Rico_Training(model,Batch_Temp,y, LR)
         xlua.progress(numBatch, nbBatch)
      end

      print("lossTemp",lossTemp/nbBatch)
      print("Test accuracy = ",accuracy(imgs_test,truth)/nbBatch)
      
   end
   save_model(model,name_save)

   
end --*

local LR=0.0001
local dataAugmentation=true
local Log_Folder='./Log/'
local list_folders_images, list_txt=Get_HeadCamera_HeadMvt()
local loading = true


image_width=200
image_height=200

torch.manualSeed(123)

model=getModel()
model=model:cuda()
parameters,gradParameters = model:getParameters()

train_Epoch(model,list_folders_images,list_txt,Log_Folder,use_simulate_images,LR)
