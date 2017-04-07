require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'optim'

require 'Get_Baxter_Files'
require 'functions'

local function ReprFromImgs(imgs,name)

   local fileName = 'preload_folder/'..'allReprSaved'..name..'.t7'
   
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

local function HeadPosFromTxts(txts, isData)
   --Since i use this function for creating X tensor for debugging
   -- or y tensor, the label tensor, i need a flag just to tell if i need X or y

   --isData = true => X tensor      isData = false => y tensor
   
   T = {}
   for l, txt in ipairs(txts) do
      truth = getTruth(txt)
      for i, head_pos in ipairs(truth) do
         T[#T+1] = head_pos
      end
   end

   T = torch.Tensor(T)

   if isData then --is it X or y that you need ?
      Ttemp = torch.zeros(T:size(1),1)
      Ttemp[{{},1}] = T
      T = Ttemp
   end

   return T
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
   batch = batch:cuda()
   y_temp = y_temp:cuda()
   return batch, y_temp
   
end

function Rico_Training(model,batch,y,reconstruct, LR)

   local criterion
   local optimizer = optim.adam

   if reconstruct then
      criterion = nn.SmoothL1Criterion():cuda()
   else
      criterion = nn.CrossEntropyCriterion():cuda()
   end
   
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

function accuracy_reconstruction(X_test,y_test, model)
   local acc = 0
   local yhat = model:forward(X_test:cuda())

   -- print("yhat",yhat[1][1],yhat[2][1],yhat[3][1],yhat[4][1],yhat[60][1])
   -- print("y",truth[1],truth[2],truth[3],truth[4],truth[60])
   
   for i=1,X_test:size(1) do
      acc = acc + math.sqrt(math.pow(yhat[i][1]-y_test[i],2))
   end
   return acc
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

function createModelReconstruction()
   net = nn.Sequential()
   net:add(nn.Linear(1,1))
   return net:cuda()
end


function train(X,y, reconstruct)
   reconstruct = reconstruct or true

   local nbList = 10
   local numEx = X:size(1)
   local splitTrainTest = 0.75

   local sizeTest = math.floor(numEx/nbList)
   
   id_test = {{math.floor(numEx*splitTrainTest), numEx}}
   X_test = X[id_test]
   y_test = y[id_test]
   
   id_train = {{1,math.floor(numEx*splitTrainTest)}}
   X_train = X[id_train]
   y_train = y[id_train]

   if reconstruct then
      model = createModelReconstruction()
      print("Test accuracy before training",accuracy_reconstruction(X_test,y_test,model))

   else
      model = createModelReward()
      print("Test accuracy before training",accuracy(X_test,y_test,model))
      print("Random accuracy", rand_accuracy(y_test))
   end
   parameters,gradParameters = model:getParameters()

   for epoch=1, NB_EPOCH do

      local lossTemp=0

      for numBatch=1, NB_BATCH do
         batch_temp, y = RandomBatch(X_train,y_train,SIZE_BATCH)
         lossTemp = lossTemp + Rico_Training(model,batch_temp,y, reconstruct, LR)
      end

      if epoch==NB_EPOCH then
         print("lossTemp",lossTemp/NB_BATCH)

         if reconstruct then
            print("Test accuracy = ",accuracy_reconstruction(X_test,y_test,model))
         else
            print("Test accuracy = ",accuracy(X_test,y_test,model))
         end
      end
   end
end

MODEL_PATH = 'Log/'

MODEL_NAME, name = 'Save97Win/reprLearner1d.t7', '97'
--MODEL_NAME,name = 'reprLearner1dWORKS.t7', 'works'
--MODEL_NAME,name = 'reprLearner1d.t7', 'shit2'

PATH_RAW_DATA = 'moreData/'
PATH_PRELOAD_DATA = 'preload_folder/'
DATA = PATH_PRELOAD_DATA..'imgsCv1.t7'

SIZE_BATCH=60
NB_EPOCH=100
LR=0.01
PLOT = true
LOADING = true

TASK = 2
reconstructingTask = true

local imgs = torch.load(DATA)
imgs[1], imgs[#imgs] = imgs[#imgs], imgs[1] -- Because during database creation we swapped those values

local _, list_txt=Get_HeadCamera_HeadMvt(PATH_RAW_DATA)

if reconstructingTask then
   y = HeadPosFromTxts(list_txt,false)
else
   y = RewardsFromTxts(list_txt)
end

--X = HeadPosFromTxts(list_txt,true)
X = ReprFromImgs(imgs, name)

NB_BATCH=math.floor(X:size(1)/SIZE_BATCH)

train(X,y,reconstructingTask)
