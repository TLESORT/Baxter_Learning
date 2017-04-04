require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'optim'

require 'Get_Baxter_Files'
require 'functions'

local function ReprFromImgs(imgs,name)

   local fileName = 'allReprSaved'..name..'.t7'
   
   if file_exists(fileName) then
      return torch.load(fileName)
   end
   
   X = {}
   local model = torch.load(MODEL_PATH..MODEL_NAME)
   for numSeq,seq in ipairs(imgs) do
      print("numSeq",numSeq)
      for i,img in ipairs(seq) do
         x = nn.utils.addSingletonDimension(img)
         X[#X+1] = model:forward(x)[1]
      end
   end
   Xtemp = torch.Tensor(X)
   X = torch.zeros(#X,1)
   X[{{},1}] = Xtemp
   torch.save(fileName,X)
   return X
end

local function HeadPosFromTxts(txts)
   X = {}
   for l, txt in ipairs(txts) do
      truth = getTruth(txt)
      for i, head_pos in ipairs(truth) do
         X[#X+1] = head_pos
      end
   end

   Xtemp = torch.Tensor(X)
   X = torch.zeros(#X,1)
   X[{{},1}] = Xtemp
   return X
end


local function RewardsFromTxts(txts)
   y = {}
   if TASK==2 then
      for l, txt in ipairs(txts) do
         truth = getTruth(txt)
         for i, head_pos in ipairs(truth) do
            if head_pos < 0.1 and head_pos > -0.1 then
               y[#y+1] = 1
            else
               y[#y+1] = 2
            end
         end
      end
   end
   
   return torch.Tensor(y)
end

local function RandomBatch(X,y,sizeBatch)

   local numSeq = X:size(1)
   batch = torch.zeros(sizeBatch,1)
   y_temp = torch.zeros(sizeBatch)

   for i=1,sizeBatch do
      local id=torch.random(1,numSeq)
      batch[{i,1}] = X[{id,1}]
      y_temp[i] = y[id] 
   end

   -- print("batch",batch)
   -- print("y_temp",y_temp)
   -- io.read()
   
   for i=1,sizeBatch do
      if y_temp[i]==1 then
         batch = batch:cuda()
         y_temp = y_temp:cuda()
         return batch, y_temp
      end
   end

   return RandomBatch(X,y,sizeBatch)
      
end

function Rico_Training(model,batch,y,LR)
   local LR=LR or 0.001
   local optimizer = optim.adam
   local criterion = nn.CrossEntropyCriterion():cuda()
   
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

      local grad = criterion:backward(yhat,y)
      model:backward(batch, grad)
      
      return loss,gradParameters
   end
   
   optimState={learningRate=LR}
   parameters, loss=optimizer(feval, parameters, optimState)
   return loss[1]
end

function accuracy(X_test,y_test,model)
   local acc = 0
   local yhat = model:forward(X_test:cuda())

   _,yId = torch.max(yhat,2)
   for i=1,X_test:size(1) do
      if yId[i][1]==y_test[i] then
         acc = acc + 1
      end
   end
   return acc/y_test:size(1)
end

function rand_accuracy(y_test)
   count = 0
   for i=1,y_test:size(1) do
      if y_test[i]==2 then
         count = count + 1
      end
   end
   return count/y_test:size(1)
end

function createModelReward()
   net = nn.Sequential()
   net:add(nn.Linear(1,3))
   net:add(nn.Tanh())
   net:add(nn.Linear(3,2))
   return net:cuda()
end

function createModelReward()
   net = nn.Sequential()
   net:add(nn.Linear(1,3))
   net:add(nn.Tanh())
   net:add(nn.Linear(3,1))
   return net:cuda()
end

function trainRewardPrediction(X,y)
   local sizeBatch=80
   local nbEpoch=100
   local LR=0.05

   local nbBatch=math.floor(X:size(1)/sizeBatch)

   local PLOT = true
   local LOADING = true

   local nbList = 10
   local numEx = X:size(1)

   local sizeTest = math.floor(numEx/nbList)
   
   id_test = {{math.floor(numEx*0.9), numEx}}
   X_test = X[id_test]
   y_test = y[id_test]
   
   id_train = {{1,math.floor(numEx*0.9)}}
   X_train = X[id_train]
   y_train = y[id_train]

   model = createModelReward()
   parameters,gradParameters = model:getParameters()

   print("Test accuracy before training",accuracy(X_test,y_test,model))
   print("Random accuracy", rand_accuracy(y_test))
   for epoch=1, nbEpoch do

      print('--------------Epoch : '..epoch..' ---------------')
      local lossTemp=0

      for numBatch=1, nbBatch do
         batch_temp, y = RandomBatch(X_train,y_train,sizeBatch)

         lossTemp = lossTemp + Rico_Training(model,batch_temp,y, LR)
         xlua.progress(numBatch, nbBatch)
      end
      print("lossTemp",lossTemp/nbBatch)
      print("Test accuracy = ",accuracy(X_test,y_test,model))
   end
end


function trainRewardPrediction(X,y)

   local nbList = 10
   local numEx = X:size(1)

   local sizeTest = math.floor(numEx/nbList)
   
   id_test = {{math.floor(numEx*0.9), numEx}}
   X_test = X[id_test]
   y_test = y[id_test]
   
   id_train = {{1,math.floor(numEx*0.9)}}
   X_train = X[id_train]
   y_train = y[id_train]

   model = createModelReconstruction()
   parameters,gradParameters = model:getParameters()

   print("Test accuracy before training",accuracy(X_test,y_test,model))
   print("Random accuracy", rand_accuracy(y_test))
   for epoch=1, NB_EPOCH do

      print('--------------Epoch : '..epoch..' ---------------')
      local lossTemp=0

      for numBatch=1, NB_BATCH do
         batch_temp, y = RandomBatch(X_train,y_train,SIZE_BATCH)

         lossTemp = lossTemp + Rico_Training(model,batch_temp,y, LR)
         xlua.progress(numBatch, NB_BATCH)
      end
      print("lossTemp",lossTemp/NB_BATCH)
      print("Test accuracy = ",accuracy(X_test,y_test,model))
   end
end

MODEL_PATH = 'Log/'
--MODEL_NAME, name = 'Save97Win/reprLearner1d.t7', '97'
MODEL_NAME,name = 'reprLearner1dWORKS.t7', 'works'
PATH_RAW_DATA = 'moreData/'
PATH_PRELOAD_DATA = 'preload_folder/'
DATA = PATH_PRELOAD_DATA..'imgsCv1.t7'

SIZE_BATCH=80
NB_EPOCH=100
LR=0.05
PLOT = true
LOADING = true

TASK = 2

local NB_BATCH=math.floor(X:size(1)/SIZE_BATCH)


local imgs = torch.load(DATA)
imgs[1], imgs[#imgs] = imgs[#imgs], imgs[1] -- Because during database creation we swapped those values

local _, list_txt=Get_HeadCamera_HeadMvt(PATH_RAW_DATA)

y = RewardsFromTxts(list_txt)
X = ReprFromImgs(imgs, name)
--X = HeadPosFromTxts(list_txt)

--trainRewardPrediction(X,y)
trainReconstruction(X,y)
