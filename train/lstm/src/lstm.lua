
local lstm = torch.class('lstm')

function lstm:__init(mode, opt)
  -- mode can be dev or prod
  self.mode = mode
  self.opt = opt
  -- self.thisopt = nil -- this will be popoulated with the opt for this experiment, in case we're reading an earlier, existing model

  self:require()
  self:set_opts()
  self:init_gpu()

  self.protos = {}
  self.params = nil
  self.grad_params = nil
  self.clones = {}
  self.dataset = nil

  self:setup_model() -- needs change

end


function lstm:require()
  require 'torch'
  require 'nn'
  require 'nngraph'
  require 'src.res'
  model_utils = require 'src.model_utils' -- note: this is global for now
  if not LSTMsq then
    LSTMsq = require 'src.LSTMsq' -- likewise, in global namespace
  end
end


function lstm:set_opts()
  -- manual defaults for some opts parameters
  if self.opt.learning_rate_decay_after==-1 then
      self.opt.learning_rate_decay_after = self.opt.max_epochs/2
  end
  torch.manualSeed(self.opt.seed)
  -- torch.setdefaulttensortype('torch.FloatTensor')
end

function lstm:assert_opts(opt_assert)
  if not opt_assert then return end

  for k,v in pairs(opt_assert) do
    if type(v)=='string' then
      assert(v==self.opt[k])
    elseif type(v)=='number' then
      assert(math.abs(v-self.opt[k]) < 1e-5)
    else
      error('type not implemented for assertion!')
    end
  end
end


function lstm:init_gpu()
  if self.opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
      print('using CUDA on GPU ' .. self.opt.gpuid .. '...')
      cutorch.setDevice(self.opt.gpuid + 1) -- note +1 to make it
                                            -- 0 indexed! sigh lua
      cutorch.manualSeed(self.opt.seed)
    else
      print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
      print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
      print('Falling back on CPU mode')
      self.opt.gpuid = -1 -- overwrite user setting
    end
  end
end


function lstm:make_zero_state(batch_size)
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


function lstm:build_model()
  local protos = {}
  protos.enc = LSTMsq.lstm(self.opt.input_size, self.opt.rnn_size, self.opt.num_layers, 'raw', nil, self.opt.dropout, self.opt.drop_inputs)
  protos.dec = LSTMsq.lstm(self.opt.input_size, self.opt.rnn_size, self.opt.num_layers, 'raw', 'sigmoid', self.opt.dropout, self.opt.drop_inputs)
  protos.dec_criterion = nn.BCECriterion()
  return protos
end


function lstm:setup_model()
  -- if an existing model is read through self.opt.init_from,
    -- current experiment's self.opt is renamed to self.thisopt and self.opt is overwritten the read model's experiment details (except for gpuid)
  local do_random_init = true
  local params, grad_params, protos

  if string.len(self.opt.init_from) > 0 then
    print('loading an LSTM from checkpoint ' .. self.opt.init_from)
    local checkpoint = torch.load(self.opt.init_from)
    protos = checkpoint.protos
    -- take opt from previous experiment, and override gpuid on prev expt settings
    assert(self.thisopt==nil, 'trying to read the model twice!')
    self.thisopt = self.opt
    self.opt = checkpoint.opt
    self.opt.gpuid = self.thisopt.gpuid
    do_random_init = false
    self:assert_opts(self.thisopt.opt_assert) -- make sure some opts are the same when read
  else
    protos = self:build_model()
  end

  self:ship_model_gpu()

  -- put the above things into one flattened parameters tensor
  print('flattening parameters..')
  if opt.use_h_init then
      params, grad_params = model_utils.combine_all_parameters(protos.enc, protos.dec, protos.h_init)
  else
      params, grad_params = model_utils.combine_all_parameters(protos.enc, protos.dec)
  end

  -- initialization
  if do_random_init then
    params:uniform(-0.08, 0.08) -- small uniform numbers
    if opt.forget_bias_init>0 then
      self:set_forget_bias(protos.enc, opt.num_layers, opt.forget_bias_init)
      self:set_forget_bias(protos.dec, opt.num_layers, opt.forget_bias_init)
    end
  end

  print('number of parameters in the model: ' .. params:nElement())
  self.protos = protos
  self.params = params
  self.grad_params = grad_params
  self.clones = self:make_clones(self.protos, self.opt.max_seqlen+2) -- plus 2 because we're adding _eos_ to start and end of sequences
end

function lstm:make_clones(protos, T)
  -- make a bunch of clones after flattening, as that reallocates memory
  local clones = {}
  for name,proto in pairs(protos) do
      if name~='h_init' then -- dont need to clone h_init across time
          print('cloning ' .. name)
          clones[name] = model_utils.clone_many_times(proto, T)
      else
          print('NOT cloning ' .. name)
      end
  end
  return clones
end


function lstm:set_forget_bias(module, num_layers, bias)
  -- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
  for layer_idx = 1, num_layers do
    for _,node in ipairs(module.forwardnodes) do
      if node.data.annotations.name == "i2h_" .. layer_idx or node.data.annotations.name == "h2h_" .. layer_idx then
        print(string.format('setting forget gate biases to %.2f in LSTM layer %s', bias/2, node.data.annotations.name))
        -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
        node.data.module.bias[{{opt.rnn_size+1, 2*opt.rnn_size}}]:fill(bias/2)
      end
    end
  end
end


function lstm:ship_model_gpu()
  -- ship the model to the GPU if desired
  if self.opt.gpuid >= 0 then
    print('shipping model to GPU..')
    for k,v in pairs(self.protos) do v:cuda() end
  end
end


function lstm:predict(data)
  local s1,m1,k1 = unpack(data)
  local n1 = s1:size(3)
  local om1 = torch.ones(m1:size())-m1
  local new_batch_size = s1:size(1)

  local range = torch.range(1,new_batch_size):type('torch.LongTensor')
  if self.opt.gpuid >= 0 then -- ship the input arrays to GPU
      -- if you have integer arrays, convert to float first and then cuda, like :float():cuda() because integers can't be cuda()'d
      s1 = s1:float():cuda()
      m1 = m1:cuda()
      om1 = om1:cuda()
      range = range:cuda()
      if k1 then k1 = k1:cuda() end
  end

  ------------------- forward pass -------------------
  local init_state = self:make_zero_state(new_batch_size)
  local rnn_state = {[0] = init_state}
  local preds

  for t=1,n1 do
    self.clones.dec[t]:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)
    local s1_t = s1[{{}, {}, t}]
    local input = {s1_t, m1[{{}, t}], om1[{{}, t}], unpack(rnn_state[t-1])}
    local lst = self.clones.dec[t]:forward(input)
    rnn_state[t] = {}
    for i=1,2*self.opt.num_layers do table.insert(rnn_state[t], lst[i]) end -- extract the state

    if t==n1 then
      preds = lst[#lst] -- last element is the prediction
    end
  end

  if preds[1][1] ~= preds[1][1] then
    print('prediction is nan!')
    mobdebug.start()
    local tmpa=1
  end

  -- compute course recommendation ranking evaluation for val/test
  preds[torch.ge(k1,1)] = -1 -- disregard predictions for those courses that have been taken during training
  local y,ind = torch.sort(preds,2,true)

  -- re ship from GPU to CPU because nonzero() is not support on GPU yet
  y = y:float()
  ind = ind:float()

  return y,ind
end

-- function lstm:optimize()
--   -- start optimization here
--   local train_losses = {}
--   local train_res = res_avg()
--   local train_res_print_every = res_avg()
--   local time_avg = res()
--   local val_losses = {}
--   local val_perfs = {}
--   local val_iters = {}
--   local test_perfs = {}
--   local test_perfs_fnames = {}
--   local checkpoint_fnames = {}
--   local optim_config, optim_state
--   if opt.optimizer=='rmsprop' then
--     optim_config = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
--   elseif opt.optimizer=='adagrad' then
--     optim_config = {learningRate = opt.learning_rate}
--   elseif opt.optimizer=='sgd' then
--     optim_config = {learningRate = opt.learning_rate, momentum = opt.momentum}
--   elseif opt.optimizer=='adam' then
--     optim_config = {learningRate = opt.learning_rate}
--   else
--     print('undefined optimizer')
--   end

--   local iterations_per_epoch
--   if opt.seq_mode=='set' then
--       iterations_per_epoch = math.ceil(ds:get_num_data('train')/opt.batch_size)
--   elseif opt.seq_mode=='permute' then
--       iterations_per_epoch = math.ceil(ds:get_num_data('train'))
--   end
--   local iterations = math.ceil(opt.max_epochs * iterations_per_epoch)
--   local unfinished = true
--   local i = 0
--   if opt.eval_val_every<1 then
--       opt.eval_val_every = math.ceil(opt.eval_val_every * iterations_per_epoch)
--   end
--   if opt.print_every<1 then
--       opt.print_every = math.ceil(opt.print_every * iterations_per_epoch)
--   end
--   local loss0_min_iter = 500
--   local loss0_avg = res()
--   local loss0_const
--   if opt.seq_mode=='set' then
--       loss0_const = 3
--   elseif opt.seq_mode=='permute' then
--       loss0_const = 3*4
--   end

--   while unfinished do
--       i = i + 1
--       local epoch = i / iterations_per_epoch
--       local timer = torch.Timer()
--       local _, loss
--       if opt.optimizer=='rmsprop' then
--           _, loss = optim.rmsprop(feval, params, optim_config)
--           -- _, loss, optim_state = optim.rmsprop(feval, params, optim_config, optim_state)
--       elseif opt.optimizer=='adagrad' then
--           _, loss = optim.adagrad(feval, params, optim_config)
--       elseif opt.optimizer=='sgd' then
--           _, loss = optim.sgd(feval, params, optim_config)
--       elseif opt.optimizer=='adam' then
--           _, loss = optim.adam(feval, params, optim_config)
--       else
--           print('undefined optimizer')
--       end
--       timer:stop()
--       time_avg:update_avg(timer:time().real)

--       local train_loss = loss[1] -- the loss is inside a list, pop it
--       train_losses[i] = train_loss
--       if global_scan_complete then
--           print('finished one compelte scan over training set.. resetting train_loss(avg) and continuing..')
--           train_res = res_avg()
--       end
--       train_res:update_avg(train_loss, global_train_num_samp)
--       train_res_print_every:update_avg(train_loss, global_train_num_samp)

--       -- exponential learning rate decay
--       if i % iterations_per_epoch == 0 and opt.learning_rate_decay < 1 then
--           if epoch >= opt.learning_rate_decay_after then
--               local decay_factor = opt.learning_rate_decay
--               optim_config.learningRate = optim_config.learningRate * decay_factor -- decay it
--               print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_config.learningRate)
--           end
--       end

--       if i % opt.print_every == 0 then
--           print(string.format("%d/%d (epoch %.5f), train_loss(m1)[m2] = %6.8f (%6.8f) [%6.8f], grad/param norm = %6.4e, time/batch = %.2fs",
--                 i, iterations, epoch, train_loss, train_res_print_every.avg, train_res.avg, grad_params:norm() / params:norm(), time_avg.avg))
--           train_res_print_every = res_avg()
--           time_avg = res()
--       end
--       -- every now and then or on last iteration
--       if i % opt.eval_val_every == 0 or i == iterations then -- or i % iterations_per_epoch == 0 then
--           -- evaluate loss on validation data
--           local val_loss, rankperf = evaluate('val')
--           val_losses[#val_losses+1] = val_loss
--           val_iters[#val_iters+1] = i

--           local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
--           print('saving checkpoint to ' .. savefile)
--           local checkpoint = {}
--           checkpoint.protos = protos
--           checkpoint.opt = opt
--           checkpoint.train_losses = train_losses
--           checkpoint.val_loss = val_loss
--           checkpoint.val_losses = val_losses
--           checkpoint.val_iters = val_iters
--           checkpoint.i = i
--           checkpoint.epoch = epoch
--           rankperf:update_ebar()
--           print(string.format("%d/%d (epoch %.5f), val_loss = %6.8f, rankperf = %6.8f +/- %6.8f", i, iterations, epoch, val_loss, rankperf.avg, rankperf.conf_intr))
--           val_perfs[#val_perfs+1] = rankperf

--           local test_loss, rankperf = evaluate('test')
--           rankperf:update_ebar()
--           print(string.format("%d/%d (epoch %.5f), TEST_loss = %6.8f, rankperf = %6.8f +/- %6.8f", i, iterations, epoch, test_loss, rankperf.avg, rankperf.conf_intr))
--           test_perfs[#test_perfs+1] = rankperf

--           local test_fname = string.format('testperf_%s_epoch%.2f.txt',opt.savefile,epoch)
--           rankperf:write_json(string.format('res/%s',test_fname))
--           test_perfs_fnames[#test_perfs_fnames+1] = test_fname

--           checkpoint.test_perfs = test_perfs
--           checkpoint.test_perfs_fnames = test_perfs_fnames
--           torch.save(savefile, checkpoint)
--           checkpoint_fnames[#checkpoint_fnames+1] = savefile

--           -- plot the learning curve
--           local xrange = torch.range(1,#checkpoint.train_losses)
--           gnuplot.figure(1)
--           gnuplot.pngfigure(string.format('res/curve_%s.png',opt.savefile))
--           gnuplot.title('RMSprop loss over iterations')
--           gnuplot.xlabel('iterations')
--           gnuplot.ylabel('Loss (neg log likelihood)')
--           gnuplot.plot({'train',xrange, torch.Tensor(checkpoint.train_losses), '~'},
--                        {'val',torch.Tensor(checkpoint.val_iters), torch.Tensor(checkpoint.val_losses), '~'})
--           gnuplot.plotflush()

--           if i==iterations then -- finishing training, find the best validation score and perform test error
--               local min_val_perf = 1e10
--               local best_ind = -1
--               for j=1,#val_perfs do
--                   if val_perfs[j].avg<min_val_perf then
--                       min_val_perf = val_perfs[j].avg
--                       best_ind = j
--                   end
--               end
--               if (#val_perfs - best_ind) < 0.1 * #val_perfs  then
--                   -- best validation is close to max_epochs, increase epoch by 2 more
--                   print('best validation too close to max_epochs, increasing max_epochs by 2 and continuing..')
--                   opt.max_epochs = opt.max_epochs + 2
--                   iterations = math.ceil(opt.max_epochs * iterations_per_epoch)
--               else
--                   unfinished = false

--                   local oldfname = checkpoint.test_perfs_fnames[best_ind]
--                   print(string.format('best validation perf of %6.8f was found at index %d/%d corresponding to test loss file %s',
--                                       min_val_perf, best_ind, #val_perfs, oldfname))
--                   -- local luatab = read_json(oldfname)
--                   local newfname = string.format('final_%s',oldfname)
--                   os.execute(string.format('cp res/%s res/%s',oldfname,newfname))

--                   local to_write = {
--                       best_val_perf = val_perfs[best_ind],
--                       best_test_perf = test_perfs[best_ind],
--                       opt = opt,
--                       checkpoint_fname = checkpoint_fnames[best_ind]
--                   }
--                   write_json(string.format('res/%s.txt', opt.savefile), to_write)
--               end
--           end
--       end

--       if i % 100 == 0 then collectgarbage() end

--       -- handle early stopping if things are going really bad
--       if loss[1] ~= loss[1] then
--           print([[loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.
--                 Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?]])
--           break -- halt
--       end
--       if i < loss0_min_iter then
--           loss0_avg:update_avg(loss[1])
--       else
--           if loss0 == nil then loss0 = loss0_avg.avg end
--           if loss[1] > loss0 * loss0_const then
--               print(string.format('loss is exploding to %.2f from loss0 of %.2f at iter %d, aborting.', loss[1], loss0, i))
--               mobdebug.start()
--               local tmpa=1
--               break -- halt
--           end
--       end
--   end
-- end

