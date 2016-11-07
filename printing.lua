---------------------------------------------------------------------------------------
-- Function : 
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function Print_performance(Models,imgs,txt, name, Log_Folder, use_simulate_images,truth)

	local REP_criterion=get_Rep_criterion()
	local PROP_criterion=get_Prop_criterion()
	local CAUS_criterion=get_Caus_criterion()
	local TEMP_criterion=nn.MSDCriterion()

	local Temp=0
	local Rep=0
	local Prop=0
	local Caus=0
	local Model=Models.Model1

	local list_out1={}

	for i=1, #imgs do
		image1=imgs[i]
		Data1=torch.Tensor(1,3,200,200)
		Data1[1]=image1
		
		Model:forward(Data1:cuda())
		local State1=Model.output[1]	

		table.insert(list_out1,State1)
	end

	-- biased estimation of test loss
	local nb_sample=100

	for i=1, nb_sample do
		Prop_batch=getRandomBatch(imgs, txt, 1, 200, 200, 'Prop', use_simulate_images)
		Temp_batch=getRandomBatch(imgs, txt, 1, 200, 200, 'Temp', use_simulate_images)
		Caus_batch=getRandomBatch(imgs, txt, 1, 200, 200, 'Caus', use_simulate_images)
		
		Temp=Temp+doStuff_temp(Models,TEMP_criterion, Temp_batch)
		Prop=Prop+doStuff_Prop(Models,PROP_criterion,Prop_batch)	
		Caus=Caus+doStuff_Caus(Models,CAUS_criterion,Caus_batch)
		Rep=Rep+doStuff_Rep(Models,REP_criterion,Prop_batch)
	end

	corr=ComputeCorrelation(truth,list_out1,1)
	show_figure(list_out1, Log_Folder..'state'..name..'.log')
	print("Ref Mutual information")
	print("truth : "..mutual_information(truth,truth))
	print("estimate : "..mutual_information(list_out1,list_out1))
	print("Mutual information")
	print(mutual_information(truth,list_out1))

	return Temp/nb_sample,Prop/nb_sample, Rep/nb_sample, Caus/nb_sample, corr
end

function ComputeCorrelation(truth,output,dimension)
	Truth=torch.Tensor(#truth)
	Output=torch.Tensor(#output)
	for i=1, #truth do
			Truth[i]=truth[i]
			Output[i]=output[i]
	end
	corr=torch.Tensor(dimension)

	for i=1,dimension do
		corr[i]=torch.cmul((Truth-Truth:mean()),(Output-Output:mean())):mean()
		corr[i]=corr[i]/(Truth:std()*Output:std())
	end

	print("Correlation")
	print(corr[1])
	if corr[1]~=corr[1] then corr[1]=0 end
	return corr:clone()[1]
end

function mutual_information(Real, Estimate)
	Real=torch.Tensor(Real)
	Estimate=torch.Tensor(Estimate)
	local real=torch.floor(Real:clone()*1000)/1000
	local estimate=torch.floor(Estimate:clone()*1000)/1000
	local division=10

	local pas_real=(real:max()-real:min())/division
	local pas_estimate=(estimate:max()-estimate:min())/division


	local prob_real=torch.zeros(division)
	local prob_estimate=torch.zeros(division)
	local prob_both=torch.zeros(division,division)

	for i=1 , real:size(1) do
		for j=1, division do
			if real[i]<=(j*pas_real+real:min()) and real[i]>=((j-1)*pas_real+real:min())  then x_real=j end
			if estimate[i]<=(j*pas_estimate+estimate:min()) and estimate[i]>=((j-1)*pas_estimate+estimate:min()) then x_estimate=j end
		end
		prob_real[x_real]=prob_real[x_real]+1
		prob_estimate[x_estimate]=prob_estimate[x_estimate]+1
		prob_both[x_real][x_estimate]=prob_both[x_real][x_estimate]+1
	end

	prob_real=prob_real/real:size(1)
	prob_estimate=prob_estimate/real:size(1)
	prob_both=prob_both/real:size(1)


	local mutual_info=0
	for x=1 , division do
	for x2=1 , division do
		if prob_real[x]*prob_estimate[x2]*prob_both[x][x2] ~= 0 then
			mutual_info=mutual_info+prob_both[x][x2]*math.log(prob_both[x][x2]/(prob_real[x]*prob_estimate[x2]))
		end

	end
	end
	return mutual_info

end

---------------------------------------------------------------------------------------
-- Function : Print_Grad(Temp_grad_list,Prop_grad_list,Rep_grad_list,Caus_grad_list)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function Print_Grad(Temp_grad_list,Prop_grad_list,Rep_grad_list,Caus_grad_list,Log_Folder)

	local Name = Log_Folder..'Grad.log'
	local accLogger = optim.Logger(Name)

	for i=1, #Temp_grad_list do
	-- update logger
		accLogger:add{['Temp_Grad'] = Temp_grad_list[i],
				['Prop_Grad'] = Prop_grad_list[i],
				['Rep_Grad'] = Rep_grad_list[i],
				['Caus_Grad'] = Caus_grad_list[i]}
	end
	-- plot logger
	accLogger:style{['Temp_Grad'] = '-',
			['Prop_Grad'] = '-',
			['Rep_Grad'] = '-',
			['Caus_Grad'] = '-'}
	accLogger.showPlot = false
	accLogger:plot()
end

---------------------------------------------------------------------------------------
-- Function : Print_Loss(Temp_Train,Prop_Train,Rep_Train,Temp_Test,Prop_Test,Rep_Test,Log_Folder)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function Print_Loss(Temp_Train,Prop_Train,Rep_Train,Caus_Train,Temp_Test,Prop_Test,Rep_Test,Caus_Test,Log_Folder)
	show_loss(Temp_Train,Temp_Test, Log_Folder..'Temp-loss.log')
	show_loss(Prop_Train,Prop_Test, Log_Folder..'Prop-loss.log')
	show_loss(Rep_Train,Rep_Test, Log_Folder..'Rep-loss.log')
	show_loss(Caus_Train,Caus_Test, Log_Folder..'Caus-loss.log')
	
	show_Allloss(Temp_Train,Prop_Train,Rep_Train,Caus_Train, Log_Folder..'train-Allloss.log')
	show_Allloss(Temp_Test,Prop_Test,Rep_Test,Caus_Test, Log_Folder..'test-Allloss.log')
end


---------------------------------------------------------------------------------------
-- Function : show_loss(list_train, list_test, Name , scale)
-- Input (list_train): list of the train loss
-- Input (list_test): list of the test loss
-- Input (Name): Name of the file
-- Input (scale): multiplicator factor needed because for optim.logger 1.1=1 but 11~=10
---------------------------------------------------------------------------------------
function show_loss(list_train, list_test, Name)
	-- log results to files
	local accLogger = optim.Logger(Name)

	for i=1, #list_train do
	-- update logger
		accLogger:add{['train'] = list_train[i],['test'] = list_test[i]}
	end
	-- plot logger
	accLogger:style{['train'] = '-',['test'] = '-'}
	accLogger.showPlot = false
	accLogger:plot()
end

function show_Allloss(Temp,Prop,Rep,Caus, Name)
	-- log results to files
	local accLogger = optim.Logger(Name)

	for i=1, #Temp do
	-- update logger
		accLogger:add{['Temp'] = Temp[i],['Prop'] = Prop[i],['Rep'] = Rep[i],['Caus'] = Caus[i]}
	end
	-- plot logger
	accLogger:style{['Temp'] = '-',['Prop'] = '-',['Rep'] = '-',['Caus'] = '-'}
	accLogger.showPlot = false
	accLogger:plot()
end

---------------------------------------------------------------------------------------
-- Function : show_figure(list_out1, Name , scale)
-- Input (list_out1): list of the estimate state
-- Input (Name) : Name of the file
-- Input (scale) : multiplicator factor needed because for optim.logger 1.1=1 but 11~=10
---------------------------------------------------------------------------------------
function show_figure(list_out1, Name , Variable_Name, point)

	local Variable_Name=Variable_Name or 'out1'
	local point=point or '+'
	-- log results to files
	accLogger = optim.Logger(Name)

	for i=1, #list_out1 do
	-- update logger
		accLogger:add{[Variable_Name] = list_out1[i]}
	end
	-- plot logger
	accLogger:style{[Variable_Name] = point}
	accLogger.showPlot = false
	accLogger:plot()
end
