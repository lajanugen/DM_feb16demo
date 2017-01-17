
local lstm_prod = torch.class('lstm_prod')

function lstm_prod:__init(model_path, opt_assert, gpuid)
  self.model_path = model_path
  self.opt_assert = opt_assert
  self.gpuid = gpuid
  self.seed = 123

  self.protos = {}
  self.params = nil
  self.grad_params = nil
  self.clones = {}
  self.dataset = nil

  self:require()
  self:init_gpu()
  self:read_model()
  self:assert_opts()

  self:ship_model_gpu()
  self:flatten_params()
  self:make_clones()
end


function lstm_prod:assert_opts()
  for k,v in pairs(self.opt_assert) do
    if k~='gpuid' then
      if type(v)=='string' or type(v)=='boolean' then
        if v~='gpuid' and v~='seed' then
          assert(v==self.opt[k])
        end
      elseif type(v)=='number' then
        assert(math.abs(v-self.opt[k]) < 1e-5, string.format('%.2f and %.2f not same for option %s!', v, self.opt[k], k))
      else
        error('ERROR: type not implemented for assertion!')
      end
    end
  end
end

function lstm_prod:require()
  require 'torch'
  require 'nn'
  require 'nngraph'
  require 'src.res'
  model_utils = require 'src.model_utils' -- note: this is global for now
  -- if not LSTMsq then
  --   LSTMsq = require 'src.LSTMsq' -- in global namespace
  -- end
end


function lstm_prod:init_gpu()
  if self.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
      -- print('using CUDA on GPU ' .. self.gpuid .. '...')
      cutorch.setDevice(self.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
      cutorch.manualSeed(self.seed)
    else
      print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
      print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
      print('Falling back on CPU mode')
      self.gpuid = -1 -- overwrite user setting
    end
  end
end


function lstm_prod:make_zero_state(batch_size)
  -- the initial state of the cell/hidden states
  local init_state = {}
  for L=1,self.opt.num_layers do
    local h_init = torch.zeros(batch_size, self.opt.rnn_size)
    if self.opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(init_state, h_init:clone())
    table.insert(init_state, h_init:clone())
  end
  return init_state
end


function lstm_prod:read_model()
  -- print('loading an LSTM from checkpoint ' .. self.model_path)
  local checkpoint = torch.load(self.model_path)
  self.protos = checkpoint.protos
  self.opt = checkpoint.opt

  -- override gpuid and seed on prev expt settings
  self.opt.gpuid = self.gpuid
  self.opt.seed = self.seed
end


function lstm_prod:ship_model_gpu()
  -- ship the model to the GPU if desired
  if self.opt.gpuid >= 0 then
    -- print('shipping model to GPU..')
    for k,v in pairs(self.protos) do v:cuda() end
  end
end


function lstm_prod:flatten_params()
  -- print('flattening parameters..')
  self.params, self.grad_params = model_utils.combine_all_parameters(self.protos.enc, self.protos.dec)
  -- print('number of parameters in the model: ' .. self.params:nElement())
end


function lstm_prod:make_clones()
  -- make a bunch of clones after flattening, as that reallocates memory
  self.clones = {}
  for name,proto in pairs(self.protos) do
    if name~='h_init' then
      -- print('cloning ' .. name)
      self.clones[name] = model_utils.clone_many_times(proto, self.opt.max_seqlen)
    end
  end
end


function lstm_prod:predict(data)
    ------------------ get minibatch -------------------
    local s1_usr,s1_sys,u1 = unpack(data)
    local n1 = u1:size(2)
    local m1 = u1
    local om1 = torch.ones(m1:size()) - m1

    local new_batch_size = u1:size(1)

    assert(u1:size(1)==s1_usr:size(1))
    assert(u1:size(1)==s1_sys:size(1))
    assert(u1:size(2)==s1_usr:size(3))
    assert(u1:size(2)==s1_sys:size(3))
    assert(u1:size(1)==1, 'batch size should be 1')
    -- assert(n1%2==0, 'during prod-time, total number of turns must be even just like during training-time, starting with user turn at first')
    assert(torch.all(torch.eq(s1_sys[{{}, {}, n1}],0))==true, 'during prod-time, total number of turns must be even just like during training-time,' ..
           ' starting with user turn at first and ending with system turn at last, BUT last system turn should simply be zeros as' ..
           ' this is what we are trying to predict')

    local range = torch.range(1,new_batch_size):type('torch.LongTensor')
    if self.opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- if you have integer arrays, convert to float first and then cuda, like :float():cuda() because integers can't be cuda()'d
        s1_usr = s1_usr:float():cuda()
        s1_sys = s1_sys:float():cuda()
        u1 = u1:cuda()
        m1 = m1:cuda()
        om1 = om1:cuda()
        range = range:cuda()
    end

    ------------------- forward pass -------------------
    local zero_state = self:make_zero_state(new_batch_size)
    local init_state
    if self.opt.use_h_init then
        init_state = self.protos.h_init:forward(zero_state)
    else
        init_state = zero_state
    end
    local rnn_state = {[0] = init_state} -- state trackers
    -- local preds_mask, targets_mask, bool_mask = {}, {}, {} -- prediction mask trackers
    local turn_type = {}

    local pred
    -- encode through first lstm
    for t=1,2*n1-1 do
        local net, s1_t, s1_tp1
        local T = math.floor((t-1)/2)+1
        if t%2==1 then          -- user turn
            net = self.clones.dec[t]
            turn_type[t] = 'user'
            s1_t = s1_usr[{{}, {}, T}]
            s1_tp1 = s1_sys[{{}, {}, T}]
        else                    -- system turn
            net = self.clones.enc[t]
            turn_type[t] = 'sys'
            s1_t = s1_sys[{{}, {}, T}]
        end
        net:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)

        local input = {s1_t, m1[{{}, T}], om1[{{}, T}], unpack(rnn_state[t-1])}
        -- mobdebug.start()
        -- local tmpa=1

        local lst = net:forward(input)
        rnn_state[t] = {}
        for i=1,2*self.opt.num_layers do table.insert(rnn_state[t], lst[i]) end -- extract the state
        -- bool_mask[t] = torch.eq(m1[{{}, T}],1) --t+1 if not for this special setting where t,T exists.

        if turn_type[t] == 'user' and T==s1_usr:size(3) then
            -- targets_mask[t] = s1_tp1:index(1,range[bool_mask[t]])   --[{range[bool_mask[t]], {}}]
            pred = lst[#lst] --:index(1,range[bool_mask[t]]) -- last element is the prediction

            if pred[1][1] ~= pred[1][1] then
                print('prediction is nan!')
            end
            break
        end
    end

    pred = torch.ge(pred[1], 0.5)
    pred:resize(1, pred:size(1))
    return pred -- only one batch should exist
end

