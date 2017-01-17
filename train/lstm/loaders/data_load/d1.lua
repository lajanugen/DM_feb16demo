
local dialogs = torch.class('dialogs')
function dialogs:__init(mode,batch_size)

	self.user_acts_noarg = {'ack','affirm','bye','hello','help','negate','null','repeat','reqalts','reqmore','restart','silence','thankyou','inform'} -- added 'inform' cz it can take 'this'
	self.user_acts_arg	 = {'confirm','deny','inform','request'}
	self.user_acts = {'ack','affirm','bye','hello','help','negate','null','repeat','reqalts','reqmore','restart','silence','thankyou','confirm','deny','inform','request'}

	self.system_acts_noarg	= {'affirm','bye','canthear','confirm-domain','negate','repeat','reqmore','welcomemsg'}
	self.system_acts_arg	= {'canthelp','canthelp.missing_slot_value','expl-conf','impl-conf','inform','offer','request','select','canthelp.exception'}
	self.system_acts = {'affirm','bye','canthear','confirm-domain','negate','repeat','reqmore','welcomemsg','canthelp','canthelp.missing_slot_value','expl-conf',
											'impl-conf','inform','offer','request','select'}

	self.allacts = {'affirm','bye','canthear','confirm-domain','negate','repeat','reqmore','welcomemsg','canthelp','canthelp.missing_slot_value','expl-conf',
										'impl-conf','inform','offer','request','select','ack','hello','null','reqalts','restart','silence','thankyou','confirm','deny','canthelp.exception'}

	self.slots = {'area','food','name','pricerange','addr','phone','postcode','signature','count'}

	self.user_act_inds		= {}
	self.system_act_inds	= {}
	self.allact_inds		= {}
	self.slot_inds			= {}

	self.batch_size = batch_size
	self.format = 'B'

	local i
	for i = 1,#self.user_acts	do self.user_act_inds[self.user_acts[i]]		= i end
	for i = 1,#self.system_acts	do self.system_act_inds[self.system_acts[i]]	= i end
	for i = 1,#self.allacts		do self.allact_inds[self.allacts[i]]			= i end
	for i = 1,#self.slots		do self.slot_inds[self.slots[i]]				= i end

	self.user_actslot_inds = {}
	for i = 1,#self.user_acts_noarg do
		self.user_actslot_inds[self.user_acts_noarg[i]] = i
	end
	ct = #self.user_acts_noarg + 1
	for i = 1,#self.user_acts_arg do
		for j = 1,#self.slots do
			self.user_actslot_inds[self.user_acts_arg[i] .. self.slots[j]] = ct
			ct = ct + 1
		end
	end
	self.NUSacts = ct - 1

	self.system_actslot_inds = {}
	for i = 1,#self.system_acts_noarg do
		self.system_actslot_inds[self.system_acts_noarg[i]] = i
	end
	ct = #self.system_acts_noarg + 1
	for i = 1,#self.system_acts_arg do
		for j = 1,#self.slots do
			self.system_actslot_inds[self.system_acts_arg[i] .. self.slots[j]] = ct
			ct = ct + 1
		end
	end
	self.NSSacts = ct - 1

	--self.data = require('data')
	-- local data_path = 'data/dstc2/'
	local data_path = '/home/llajan/dstc2/'
	--local data_path = '/home/llajan/data/'
	self.train = read_file(data_path .. 'config/dstc2_train.flist')
	self.valid = read_file(data_path .. 'config/dstc2_dev.flist')
	self.test  = read_file(data_path .. 'config/dstc2_test.flist')
	self.alldata = {self.train,self.valid,self.test}

	self.dir_prefix = { data_path .. 'train/',
						data_path .. 'train/',
						data_path .. 'test/'}

	self.NUacts = #self.user_acts
	self.NSacts = #self.system_acts
	self.Nacts	= #self.allacts --self.NSacts + self.NUacts
	self.Nslots = #self.slots

	self.multacts = 0
	self.dialog_ptr = {1,1,1}

	if mode == 'debug-rand' then
		math.randomseed(os.time())
		self.debug_dialog = math.floor(math.random()*#self.train)
	elseif mode == 'debug-142' then
		self.debug_dialog = 29
	elseif mode == 'debug-first' then
		self.debug_dialog = 1
	else
		assert(mode=='all', 'unknown mode!')
	end

	self.mode = mode
	if self.mode ~= 'all' then
		self.batch_debug,_ = self:get_next_batch('train')
		self:reset_pointer('train')
	end

	self.debug_print = false
end

function dialogs:get_user_repr_size()
	return self.NUSacts
end

function dialogs:get_sys_repr_size()
	return self.NSSacts
end

function dialogs:get_repr_size()
	return 1 + self.Nacts + self.Nslots
end

function dialogs:reset_pointer(train_test_valid)
	self.dialog_ptr[self:get_id(train_test_valid)] = 1
end

--function dialogs:get_next_batch()
--	return self.get_next_dialog()
--end

function get_act_slot(dact)
	local dialog_act = dact['act']
	local slots = dact['slots']

	local slots_present = {}
	local values_present = {}

	if #slots == 1 then -- 1 arg
		if slots[1][1] == 'slot' then
			table.insert(slots_present,slots[1][2] )
			table.insert(values_present,nil)
		else
			table.insert(slots_present,slots[1][1])
			table.insert(values_present,slots[1][2])
		end
	elseif #slots >= 2 then --This happens only for dact 'canthelp'
		for i = 1,#slots do
			table.insert(slots_present,slots[i][1])
			table.insert(values_present,slots[i][2])
		end
	end
	return dialog_act, slots_present, values_present
end

function dialogs:get_id(train_test_valid)
	if train_test_valid == 'train' then
		return 1
	elseif train_test_valid == 'val' then
		return 2
	else
		return 3
	end
end

function dialogs:get_num_data(train_test_valid)
	return(#self.alldata[self:get_id(train_test_valid)])
end

function dialogs:get_next_batch(train_test_valid)
	local i, u, s
	local dialog_lengths, U, S = {},{},{}
    local tvt = self:get_id(train_test_valid)

	local scan_complete = false
	if self.dialog_ptr[tvt] + self.batch_size - 1 == #self.alldata[tvt] then scan_complete = true end

	for i = 1,self.batch_size do
		if self.format == 'A' then
			u,s,_ = self:get_next_dialog_formatA(train_test_valid)
		else
			u,s = self:get_next_dialog_formatB(train_test_valid)
		end
		table.insert(dialog_lengths, u:size(2))
		table.insert(U,u)
		table.insert(S,s)
	end

	if self.dialog_ptr[tvt] + self.batch_size - 1 > #self.alldata[tvt] then
		self.dialog_ptr[tvt] = 1
		scan_complete = true
	end

	local max_turns = math.max(unpack(dialog_lengths))
	local usr_mat, sys_mat
	local mask = torch.ones(self.batch_size,max_turns)
	for i = 1,self.batch_size do
		local append_offset = max_turns - U[i]:size(2)

		usr_mat = torch.zeros(self.batch_size,self.NUSacts,max_turns)
		usr_mat:sub(i,i,1,self.NUSacts,1,U[i]:size(2)):copy(U[i]:view(1,self.NUSacts,U[i]:size(2)))

		sys_mat = torch.zeros(self.batch_size,self.NSSacts,max_turns)
		sys_mat:sub(i,i,1,self.NSSacts,1,S[i]:size(2)):copy(S[i]:view(1,self.NSSacts,S[i]:size(2)))

	end
	return {usr_mat, sys_mat, mask}, scan_complete
end

function dialogs:mat_to_sentence(system_mat)

	dasv_table = {}
	dasv_table['dasv'] = {}
	for i = 1,#self.system_acts_noarg do
		if system_mat[i] == 1 then
			table.insert(dasv_table['dasv'],{['dialog-act'] = self.system_acts_noarg[i]})
		end
	end
	ct = #self.system_acts_noarg + 1
	for i = 1,#self.system_acts_arg do
		for j = 1,#self.slots do
			if system_mat[ct] == 1 then
				table.insert(dasv_table['dasv'],{['dialog-act'] = self.system_acts_arg[i],slot = self.slots[j]})
			end
			ct = ct + 1
		end
	end
	-- local pretty_json_text = JSON:encode_pretty(dasv_table)

	return dasv_table
end

--function dialogs:get_next_batch_formatA(train_test_valid)
function dialogs:get_next_dialog_formatA(train_test_valid)
--function dialogs:get_next_dialog()
    local tvt = self:get_id(train_test_valid)

	local user, system
	if self.mode == 'all' then
		user = read_json	(self.dir_prefix[tvt] .. self.alldata[tvt][self.dialog_ptr[tvt]] .. '/label.json')
		system = read_json	(self.dir_prefix[tvt] .. self.alldata[tvt][self.dialog_ptr[tvt]] .. '/log.json')
	else
		user = read_json	(self.dir_prefix[tvt] .. self.alldata[tvt][self.debug_dialog] .. '/label.json')
		system = read_json	(self.dir_prefix[tvt] .. self.alldata[tvt][self.debug_dialog] .. '/log.json')
	end


	local scan_complete = false
	self.dialog_ptr[tvt] = self.dialog_ptr[tvt] + 1
	if self.dialog_ptr[tvt] > #self.alldata[tvt] then
		self.dialog_ptr[tvt] = 1
		scan_complete = true
	end

	local turns = #user['turns']
	local dialog_tensor = nil
	local mask = {}
	local act_tensor
	--print(conv[1])
	for i = 1,turns do
			local user_acts	= user['turns'][i]['semantics']['json']
			local system_acts = system['turns'][i]['output']['dialog-acts']
		--if #acts > 1 then
			--print(system['turns'][i]['output']['transcript'])
			act_tensor = torch.Tensor(1 + self.Nacts + self.Nslots):fill(0)
			if dialog_tensor == nil then dialog_tensor = act_tensor
			else						 dialog_tensor = torch.cat(dialog_tensor,act_tensor,2)
			end
			table.insert(mask,1)
			for system_act = 1,#system_acts do
				act,slots,values = get_act_slot(system_acts[system_act])
				--print('system',act,slots)
				--local t
				--print(act)
				--for _,t in ipairs(slots) do
				--	print(t)
				--end

				-- Batch Tensor
				act_tensor = torch.Tensor(1 + self.Nacts + self.Nslots):fill(0)
				act_tensor[1 + self.allact_inds[act]] = 1
				for _,slot in ipairs(slots) do
					if slot and self.slot_inds[slot] then
						act_tensor[1 + self.Nacts + self.slot_inds[slot]] = 1
					end
				end
				if dialog_tensor == nil then dialog_tensor = act_tensor
				else						 dialog_tensor = torch.cat(dialog_tensor,act_tensor,2)
				end
				table.insert(mask,1)

			end
			-- EOD --
			act_tensor = torch.Tensor(1 + self.Nacts + self.Nslots):fill(0)
			act_tensor[1] = 1
			dialog_tensor = torch.cat(dialog_tensor,act_tensor,2)
			table.insert(mask,2)

			if i ~= turns then -- ignore last user turn

				if self.debug_print then
					io.write('#####################################','\n')
					io.write(i,user['turns'][i]['transcription'],'\n')
					io.write('####','\n')
					io.write(user_acts,'\n')
				end
				--print(#user_acts)
				if #user_acts == 0 then
					user_acts = {}
					user_acts[1] = {act = 'null',slots = {}}
				end
				for user_act = 1,#user_acts do
					act,slots,values = get_act_slot(user_acts[user_act])
					--print('user',act,slots)
					if self.debug_print then
						io.write('* Act ' .. tostring(user_act),'\n')
						io.write(act,'\n',slots,'\n')
					end

					-- Batch Tensor
					act_tensor = torch.Tensor(1 + self.Nacts + self.Nslots):fill(0)
					act_tensor[1 + self.allact_inds[act]] = 1
					for _,slot in ipairs(slots) do
						if slot and self.slot_inds[slot] then
							act_tensor[1 + self.Nacts + self.slot_inds[slot]] = 1
						end
					end

					if self.debug_print then
						io.write('-------------------------------------','\n')
						io.write('Tensor : ','\n')
						for i = 1,1+self.Nacts+self.Nslots do
							io.write(act_tensor[i],' ')
						end
						io.write('\nActs : ')
						for i = 1,self.Nacts do
							io.write(self.allacts[i],' ',act_tensor[1+i],'   ')
						end
						io.write('\nSlots : ')
						for i = 1,self.Nslots do
							io.write(self.slots[i],' ',act_tensor[1+self.Nacts+i],'  ')
						end
						io.write('\n-------------------------------------\n')
					end

					if dialog_tensor == nil then dialog_tensor = act_tensor
					else						 dialog_tensor = torch.cat(dialog_tensor,act_tensor,2)
					end
					table.insert(mask,0)
				end
				if self.debug_print then
					io.write('#####################################','\n')
				end
			end

		--end
	end
	dialog_tensor = dialog_tensor:reshape(1,dialog_tensor:size(1),dialog_tensor:size(2))

	if self.mode ~= 'all' and self.batch_debug then
		return self.batch_debug,scan_complete
	else
		return {dialog_tensor,torch.Tensor(mask)},scan_complete
	end
end

--function dialogs:get_next_batch_formatB(train_test_valid)
function dialogs:get_next_dialog_formatB(train_test_valid)
    local tvt = self:get_id(train_test_valid)
	local user = read_json	(self.dir_prefix[tvt] .. self.alldata[tvt][self.dialog_ptr[tvt]] .. '/label.json')
	local system = read_json	(self.dir_prefix[tvt] .. self.alldata[tvt][self.dialog_ptr[tvt]] .. '/log.json')
	--print(self.dir_prefix[tvt] .. self.alldata[tvt][self.dialog_ptr[tvt]] .. '/log.json')

	self.dialog_ptr[tvt] = self.dialog_ptr[tvt] + 1
	if self.dialog_ptr[tvt] > #self.alldata[tvt] then self.dialog_ptr[tvt] = 1 end

	local turns = #user['turns']

	assert(#user['turns'] == #system['turns'],'Unequal number of turns')
	local user_dialog_tensor = nil
	local system_dialog_tensor = nil
	--print(conv[1])
	user_dialog_tensor = torch.zeros(self.NUSacts,turns)
	system_dialog_tensor = torch.zeros(self.NSSacts,turns)
	for i = 1,turns-1 do
		local user_acts	= user['turns'][i]['semantics']['json']
		local system_acts = system['turns'][i+1]['output']['dialog-acts']

		-- DIALOG STARTS WITH USER --
		--
		--if i ~= turns then -- Ignore last user message
			user_dialog_tensor:sub(1,self.NUSacts,i,i):copy(self:get_tensor_user(user_acts))
		--end

		--if i ~= 1 then -- Ignore welcome message
			system_dialog_tensor:sub(1,self.NSSacts,i,i):copy(self:get_tensor_system(system_acts))
		--end
		if self.alldata[tvt][self.dialog_ptr[tvt]] == 'Mar13_S0A1/voip-14cb91bc48-20130327_202138' then
			local t = self:get_tensor_system(system_acts)
			print('--ref--')
			print(system_acts)
			print('--con--')
			print(self:mat_to_sentence(t))
			print('-----')
			--print(t:reshape(1,t:size(1)))
			--print(system_acts)
			--print(self.system_actslot_inds['welcomemsg'])
		end
	end
	return user_dialog_tensor,system_dialog_tensor
end

function dialogs:get_tensor_system(acts)
	local act_tensor = torch.Tensor(self.NSSacts):fill(0)
	assert(#acts > 0, "No dialog act")
	for single_act = 1,#acts do
		local act,slots,values = get_act_slot(acts[single_act])
		if #slots == 0 then
			act_tensor[self.system_actslot_inds[act]] = 1
		elseif #slots == 1 then
			--assert(act_tensor[self.system_actslot_inds[act .. slots[1]]] == 0, print(acts))--, 'Duplicate (slot,value) pair')
			--Commented out because this happens in the caes of select & canthelp.exception acts
			act_tensor[self.system_actslot_inds[act .. slots[1]]] = 1
		elseif #slots >= 2 then -- assume it is 'canthelp'
			for i = 1,#slots do
				assert(act_tensor[self.system_actslot_inds[act .. slots[i]]] == 0, 'Duplicate (slot,value) pair')
				act_tensor[self.system_actslot_inds[act .. slots[i]]] = 1
			end
		end
	end
	return act_tensor
end

function dialogs:get_tensor_sys_demo(acts)
	local act_tensor = torch.Tensor(self.NSSacts):fill(0)
	for i = 1,#acts do
		local act = acts[i]['dialog-act']
		local slot = acts[i]['slot']
		local value = acts[i]['value']
		if slot ~= '' then
			act_tensor[self.system_actslot_inds[act .. slot]] = 1
		else
			act_tensor[self.system_actslot_inds[act]] = 1
		end
	end
	return act_tensor
end

function dialogs:get_tensor_user_demo(acts)
	local act_tensor = torch.Tensor(self.NUSacts):fill(0)
	for i = 1,#acts do
		local act = acts[i]['dialog-act']
		local slot = acts[i]['slot']
		local value = acts[i]['value']
		if slot ~= '' and slot ~= nil then
			act_tensor[self.user_actslot_inds[act .. slot]] = 1
		else
			act_tensor[self.user_actslot_inds[act]] = 1
		end
	end
	return act_tensor
end

function dialogs:get_tensor_user(acts)
	local act_tensor = torch.Tensor(self.NUSacts):fill(0)
	if #acts == 0 then
		act_tensor[self.user_actslot_inds['null']] = 1
	end
	for single_act = 1,#acts do
		local act,slots,values = get_act_slot(acts[single_act])
		assert(#slots < 2, 'More than 2 slots in an act')
		if #slots == 0 then
			act_tensor[self.user_actslot_inds[act]] = 1
		elseif #slots == 1 then
			if slots[1] == 'this' then
				assert(act_tensor[self.user_actslot_inds[act]] == 0, 'Slot \'this\' issue')
				act_tensor[self.user_actslot_inds[act]] = 1
			else
				-- assert(act_tensor[self.user_actslot_inds[act .. slots[1]]] == 0, 'Duplicate (slot,value) pair')
				act_tensor[self.user_actslot_inds[act .. slots[1]]] = 1
			end
		end
		-- Verified that User does not have more than one slot per act
	end
	return act_tensor
end

function read_file(filename)
	local file = assert(io.open(filename, "r"));
	local arr = {}
	for line in file:lines() do
		table.insert (arr, line);
	end
	return arr
end

function read_json(fname)
	local f = assert(io.open(fname, "r"))
	local content = f:read("*all")
	f:close()
	local luatable = JSON:decode(content)
	return luatable
end

