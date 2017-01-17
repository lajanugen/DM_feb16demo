
--[[

Modified for seq2seq

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'os'
require 'gnuplot'

require 'src.res'
require 'src.misc'
require 'src.OneHot'
require 'loaders.data_load.dialogs_b'
local LSTMsq = require 'src.LSTMsq'
local pred_layer = require 'src.pred'
local model_utils = require 'src.model_utils'
--mobdebug = require 'mobdebug'
JSON = assert(loadfile "lib/JSON.lua")()

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train lstm NN model to predict courses')
cmd:text()
cmd:text('Options')
--primary things to change
cmd:option('-savefile','v2-h64-b1-lr2e-3','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-gpuid',2,'which gpu to use. -1 = use CPU')
cmd:option('-rnn_size', 64, 'size of LSTM internal state') -- 1792
cmd:option('-batch_size', 1, 'number of sequences to train on in parallel')
cmd:option('-learning_rate',2e-3,'learning rate for rmsprop or adagrad or sgd') -- * 3.16
cmd:option('-num_layers', 2, 'number of layers in the LSTM')

--secondary things to change
cmd:option('-data_mode', 'all', 'all or debug or nil')
cmd:option('-dropout',0.5,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-print_every',0.1,'how many steps/minibatches between printing out the loss; if this is <1, it indicates this value in epoch')
cmd:option('-eval_val_every',0.999,'every how many iterations should we evaluate on validation data? if this is <1, it indicates this value in epoch')
cmd:option('-checkpoint_dir', 'cache/dump/', 'output directory where checkpoints get written')

-- need not change any of below
cmd:option('-optimizer','rmsprop','which optimzer to choose, adagrad or rmsprop or sgd or adam')
cmd:option('-momentum',-1,'momentum for sgd')
-- data
cmd:option('-vocab_size', -1, 'size of vocabulary; not used in this experiment!')
cmd:option('-max_seqlen', 100, 'set large enough so that no errors are thrown')
-- model
cmd:option('-use_h_init', false, 'use h_init layer in the beginning')
-- training
cmd:option('-max_epochs', 20,'number of full passes through the training data')
cmd:option('-learning_rate_decay_after',-1,'in number of epochs, when to start decaying the learning rate; equals max_epochs/2 if -1')
-- optim
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-grad_clip_mode','clip','This can be norm or clip or none')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-forget_bias_init',5,'initialize forget bias to a high value to recommend remembering')
-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-num_batches_valset',-1,'limit the val set so that we can spend more time training, -1 to disable')
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
-- LSTMsq parameter, but not used anymore
cmd:option('-drop_inputs',false,'dropout applied to initial inputs as well')
cmd:text()
-- parse input params
local opt = cmd:parse(arg)
----------------------------------------------------------------------
-- define datasets
local ds = dialogs(opt.data_mode, opt.batch_size)
opt.vocab_size = -1
opt.u_repsiz = ds:get_user_repr_size()
opt.s_repsiz = ds:get_sys_repr_size()
lfs.mkdir('cache/')
lfs.mkdir('cache/dump/')
lfs.mkdir('res/')
----------------------------------------------------------------------
-- trainer
if opt.learning_rate_decay_after==-1 then
    opt.learning_rate_decay_after = opt.max_epochs/2
end
torch.manualSeed(opt.seed)
-- torch.setdefaulttensortype('torch.FloatTensor')
-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
----------------------------------------------------------------------
-- trainer
if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end
print('running experiment with options:',opt)
----------------------------------------------------------------------
-- trainer
function create_zero_state(batch_size)
  -- the initial state of the cell/hidden states
  local init_state = {}
  for L=1,opt.num_layers do
      local h_init = torch.zeros(batch_size, opt.rnn_size)
      if opt.gpuid >=0 then h_init = h_init:cuda() end
      table.insert(init_state, h_init:clone())
      table.insert(init_state, h_init:clone())
  end
  return init_state
end

function set_forget_bias(module, num_layers, bias)
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

function report_surrogates(dataset_str, set_perfs, set_loss, perf_surrogates, i, iterations, epoch, time_secs)
    for tmp_i = 1,#perf_surrogates do
        perf_surrogates[tmp_i]:update_ebar()
    end

    print(string.format("%d/%d (epoch %.5f), %s_loss = %6.8f, acc = %6.8f +/- %6.8f, precision = %6.8f +/- %6.8f, recall = %6.8f +/- %6.8f, time = %d sec",
          i, iterations, epoch, dataset_str, set_loss,
          perf_surrogates[1].avg, perf_surrogates[1].conf_intr, perf_surrogates[2].avg, perf_surrogates[2].conf_intr,
          perf_surrogates[3].avg, perf_surrogates[3].conf_intr, time_secs))
    set_perfs[#set_perfs+1] = perf_surrogates
end

----------------------------------------------------------------------
-- trainer
-- define the model: prototypes for one timestep, then clone them in time
local protos = {}
local do_random_init = true
if string.len(opt.init_from) > 0 then
  print('loading an LSTM from checkpoint ' .. opt.init_from)
  local checkpoint = torch.load(opt.init_from)
  protos = checkpoint.protos
  do_random_init = false
else
  -- sys repr is encoded into the LSTM by these model parameters
  protos.enc = LSTMsq.lstm(opt.s_repsiz, opt.rnn_size, opt.num_layers, 'raw', nil, opt.dropout, opt.drop_inputs)
  -- user repr is encoded into the LSTM by these model parameters, along with predicting next sys repr
  local pred_layer_settings = {{type='sigmoid', output_size=opt.s_repsiz}}
  protos.dec = LSTMsq.lstm(opt.u_repsiz, opt.rnn_size, opt.num_layers, 'raw', pred_layer_settings, opt.dropout, opt.drop_inputs)

  -- pred criterions
  protos.dec_criterion = nn.BCECriterion()
  -- protos.pred = pred_layer.create_layer(opt.input_size, 'sigmoid', opt.dropout)
end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    print('shipping model to GPU..')
    for k,v in pairs(protos) do v:cuda() end
end

-- put the above things into one flattened parameters tensor
print('flattening parameters..')
local params, grad_params
if opt.use_h_init then
    params, grad_params = model_utils.combine_all_parameters(protos.enc, protos.dec, protos.h_init)
else
    params, grad_params = model_utils.combine_all_parameters(protos.enc, protos.dec)
end
print('number of parameters in the model: ' .. params:nElement())

-- initialization
if do_random_init then
  params:uniform(-0.08, 0.08) -- small uniform numbers
  if opt.forget_bias_init>0 then
    set_forget_bias(protos.enc, opt.num_layers, opt.forget_bias_init)
    set_forget_bias(protos.dec, opt.num_layers, opt.forget_bias_init)
  end
end

-- make a bunch of clones after flattening, as that reallocates memory
local clones = {}
for name,proto in pairs(protos) do
    if name~='h_init' then
        print('cloning ' .. name)
        clones[name] = model_utils.clone_many_times(proto, opt.max_seqlen+2) -- plus 2 because we're adding _eos_ to start and end of second sequence
    end
end


function eval_surrogate(perf_surrogates, preds_mask, targets_mask, turn_type)
    for t=1,#turn_type do -- for each timestep
        if turn_type[t] == 'user' then
            local preds = torch.gt(preds_mask[t], 0.5)
            local T = math.floor((t-1)/2)+1
            local truths = targets_mask[t]
                -- mobdebug.start()
                -- local tmpa=1
            for i=1,preds:size(1) do
                local pred = preds[i]
                local truth = truths[i]

                local is_eq = torch.eq(pred, truth:cudaByte())

                -- full prediction accuracy
                local acc = (torch.all(is_eq) and 1) or 0
                perf_surrogates[1]:update_avg(acc)

                -- precision (of all 1s we predicted how many were correct/precise?)
                local is_eq_pred1 = is_eq[torch.eq(pred,1)]
                perf_surrogates[2]:update_avg_tensor(is_eq_pred1)

                -- recall (of all the true 1s how many did we predict/recall?)
                local is_eq_label1 = is_eq[torch.eq(truth,1)]
                perf_surrogates[3]:update_avg_tensor(is_eq_label1)
            end
        end
    end
end

function fwd_bwd_helper(mode, perf_surrogates)
    -- mode can be 'train' or 'val' or 'test'
    -- data batch repeats after one full cycle
    ------------------ get minibatch -------------------
    local data, scan_complete = ds:get_next_batch(mode)

    local s1_usr,s1_sys,u1 = unpack(data)
    local n1 = u1:size(2)
    local m1 = u1
    local om1 = torch.ones(m1:size()) - m1

    local new_batch_size = u1:size(1)

    assert(u1:size(1)==s1_usr:size(1))
    assert(u1:size(1)==s1_sys:size(1))
    assert(u1:size(2)==s1_usr:size(3))
    assert(u1:size(2)==s1_sys:size(3))

    local range = torch.range(1,new_batch_size):type('torch.LongTensor')
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- if you have integer arrays, convert to float first and then cuda, like :float():cuda() because integers can't be cuda()'d
        s1_usr = s1_usr:float():cuda()
        s1_sys = s1_sys:float():cuda()
        u1 = u1:cuda()
        m1 = m1:cuda()
        om1 = om1:cuda()
        range = range:cuda()
    end

    ------------------- forward pass -------------------
    local zero_state = create_zero_state(new_batch_size)
    local init_state
    if opt.use_h_init then
        init_state = protos.h_init:forward(zero_state)
    else
        init_state = zero_state
    end
    local rnn_state = {[0] = init_state} -- state trackers
    local preds_mask, targets_mask, bool_mask = {}, {}, {} -- prediction mask trackers
    local turn_type = {}
    local loss = 0

    local num_samples = 0
    -- encode through first lstm
    for t=1,2*n1-1 do
        -- if u1[t] == 2 then --
        --     rnn_state[t] = rnn_state[t-1]
        -- else -- when this is true, no input needs to be performed
            -- local num_user_t = torch.eq(u1[{{}, t}], 0):sum()
            -- local num_sys_t = torch.eq(u1[{{}, t}], 1):sum()
            local net, s1_t, s1_tp1
            local T = math.floor((t-1)/2)+1
            if t%2==1 then          -- user turn
                net = clones.dec[t]
                turn_type[t] = 'user'
                s1_t = s1_usr[{{}, {}, T}]
                s1_tp1 = s1_sys[{{}, {}, T}]
            else                    -- system turn
                net = clones.enc[t]
                turn_type[t] = 'sys'
                s1_t = s1_sys[{{}, {}, T}]
            end
            if mode=='train' then
                net:training() -- make sure we are in correct mode (this is cheap, sets flag)
            else
                net:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)
            end

            local input = {s1_t, m1[{{}, T}], om1[{{}, T}], unpack(rnn_state[t-1])}
            local lst = net:forward(input)
            rnn_state[t] = {}
            for i=1,2*opt.num_layers do table.insert(rnn_state[t], lst[i]) end -- extract the state

            bool_mask[t] = torch.eq(m1[{{}, T}],1) --t+1 if not for this special setting where t,T exists.

            if turn_type[t] == 'user' then
                targets_mask[t] = s1_tp1:index(1,range[bool_mask[t]])   --[{range[bool_mask[t]], {}}]
                preds_mask[t] = lst[#lst]:index(1,range[bool_mask[t]]) -- last element is the prediction

                if preds_mask[t][1][1] ~= preds_mask[t][1][1] then
                    print('prediction is nan!')
                    mobdebug.start()
                    local tmpa=1
                end

                -- calculate loss only if its user turn (as the next prediction is system's)
                local nums = bool_mask[t]:sum()
                -- mobdebug.start()
                -- local tmpa=1
                local currloss = clones.dec_criterion[t]:forward(preds_mask[t], targets_mask[t])
                loss = loss + currloss * nums
                num_samples = num_samples + nums
            end
        -- end
    end

    -- handle some corner cases during learning
    if num_samples~=0 then
        loss = loss / num_samples
    else
        -- this situation simply means bool_mask was never 1 all throughout. This would mean, dec_criterion:forward never accummulated any loss
        -- and so backprop will have no effect. we can safely ignore this case and next iteration will fix itself
        print('chk why though!')
        mobdebug.start()
        local tmpa=1
        return loss, num_samples, scan_complete
    end
    loss = loss/torch.log(2)

    -- compute surrogates for val/test
    if perf_surrogates or (mode=='val' or mode=='test') then
        eval_surrogate(perf_surrogates, preds_mask, targets_mask, turn_type)
        if mode=='val' or mode=='test' then
            return loss, num_samples, scan_complete
        end
    end

    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[2*n1-1] = create_zero_state(new_batch_size)}
    for t=2*n1-1,1,-1 do
        -- if u1[t] == 2 then
        --     drnn_state[t-1] = drnn_state[t]
        -- else
            -- error from criterion loss
            local T = math.floor((t-1)/2)+1
            if turn_type[t] == 'user' then -- backprop from criterion loss happens only for user turns (which will predict next sys turn), else zeros will be propagated
                local tmp_unmask = torch.zeros(new_batch_size, opt.s_repsiz)
                if opt.gpuid >= 0 then tmp_unmask = tmp_unmask:cuda() end
                -- backprop through loss, and softmax/linear
                local dout_t = clones.dec_criterion[t]:backward(preds_mask[t], targets_mask[t])
                tmp_unmask:indexCopy(1,range[bool_mask[t]],dout_t)
                table.insert(drnn_state[t], tmp_unmask)
            else
                -- assert(u1[t]==0, 'u1 must be one of 0,1,2')
            end

            local net, s1_t, s1_tp1
            if turn_type[t]=='user' then         -- user turn
                net = clones.dec[t]
                s1_t = s1_usr[{{}, {}, T}]
                s1_tp1 = s1_sys[{{}, {}, T}]
            elseif turn_type[t]=='sys' then     -- system turn
                net = clones.enc[t]
                s1_t = s1_sys[{{}, {}, T}]
            end

            local input = {s1_t, m1[{{}, T}], om1[{{}, T}], unpack(rnn_state[t-1])}
            local dlst = net:backward(input, drnn_state[t])

            drnn_state[t-1] = {}
            for k,v in pairs(dlst) do
                if k > 3 then -- k == 1,2,3 is gradient on x, mask, 1-mask, which we dont need
                    drnn_state[t-1][k-3] = v -- note we do k-3 for the same reason
                end
            end
        -- end
    end
    if opt.use_h_init then
        local tmp = protos.h_init:backward(zero_state, drnn_state[0])
    end

    return loss, num_samples, scan_complete
end

global_train_num_samp = 0
global_scan_complete = false
global_train_surr = {res(), res(), res()}
function feval(x)
    if x ~= params then
        print('params is not x!')
        params:copy(x)
    end
    grad_params:zero()

    ---------------- forward-backward step -------------
    local loss, num_samples, scan_complete = fwd_bwd_helper('train', global_train_surr)
    global_train_num_samp = num_samples
    global_scan_complete = scan_complete

    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    -- init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?

    -- clip gradient element-wise
    if opt.grad_clip_mode == 'clip' then
        grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    elseif opt.grad_clip_mode == 'norm' then
        -- elementwise, instead of clipping at grad_clip, change it to opt.grad_clip * grad_val / grad_norm
        error('not implemented!')
        -- local norm = grad_params:norm()/opt.batch_size
    elseif opt.grad_clip_mode == 'none' then
    end
    return loss, grad_params
end

-- evaluate the loss over a validation set
function evaluate(set)
    -- resets val_dataset pointer, and loops until scan_complete is true OR when num_batches == opt.num_batches_valset
    -- print(string.format('evaluating loss over %s set..',set))
    local result = res_avg()
    local i = 0
    local tot_samp = math.ceil(ds:get_num_data(set)/opt.batch_size)

    ds:reset_pointer(set)
    local perf_surrogates = {res(), res(), res()}
    while true do
        ---------------- forward step -------------
        local loss, num_samples, scan_complete = fwd_bwd_helper(set, perf_surrogates)
        result:update_avg(loss, num_samples)
        i = i + 1
        if i % opt.print_every == 0 then
            for tmp_i = 1,#perf_surrogates do
                perf_surrogates[tmp_i]:update_ebar()
            end
            -- print(string.format('%d/%d %s_loss = %6.8f, perf_surrogates = %6.8f +/- %6.8f',i,tot_samp,set,result.avg,perf_surrogates.avg,perf_surrogates.conf_intr))
        end
        if opt.num_batches_valset ~= -1 and i>=opt.num_batches_valset then
            break
        end
        if scan_complete then break end
    end
    return result.avg, perf_surrogates
end

-- start optimization here
local train_losses = {}

local train_res = res_avg()
local train_res_print_every = res_avg()

local time_avg = res()
local val_losses = {}
local val_perfs = {}
local val_iters = {}
local test_perfs = {}
local test_perfs_fnames = {}
local checkpoint_fnames = {}
local optim_config, optim_state
if opt.optimizer=='rmsprop' then
  optim_config = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
elseif opt.optimizer=='adagrad' then
  optim_config = {learningRate = opt.learning_rate}
elseif opt.optimizer=='sgd' then
  optim_config = {learningRate = opt.learning_rate, momentum = opt.momentum}
elseif opt.optimizer=='adam' then
  optim_config = {learningRate = opt.learning_rate}
else
  error('undefined optimizer')
end

local iterations_per_epoch = math.ceil(ds:get_num_data('train')/opt.batch_size)
local iterations = math.ceil(opt.max_epochs * iterations_per_epoch)
local unfinished = true
local i = 0
if opt.eval_val_every<1 then
    opt.eval_val_every = math.ceil(opt.eval_val_every * iterations_per_epoch)
end
if opt.print_every<1 then
    opt.print_every = math.ceil(opt.print_every * iterations_per_epoch)
end
local loss0_min_iter = 1
local loss0_avg = res()
local loss0_const = 3
local loss0

while unfinished do
    i = i + 1
    local epoch = i / iterations_per_epoch
    local timer = torch.Timer()
    local _, loss
    if opt.optimizer=='rmsprop' then
        _, loss = optim.rmsprop(feval, params, optim_config)
        -- _, loss, optim_state = optim.rmsprop(feval, params, optim_config, optim_state)
    elseif opt.optimizer=='adagrad' then
        _, loss = optim.adagrad(feval, params, optim_config)
    elseif opt.optimizer=='sgd' then
        _, loss = optim.sgd(feval, params, optim_config)
    elseif opt.optimizer=='adam' then
        _, loss = optim.adam(feval, params, optim_config)
    else
        error('undefined optimizer')
    end
    timer:stop()
    time_avg:update_avg(timer:time().real)

    local train_loss = loss[1] -- the loss is inside a list, pop it
    train_losses[i] = train_loss
    if global_scan_complete then
        print('finished one compelte scan over training set.. resetting train_loss(avg) and training surrogates, and continuing..')
        train_res = res_avg()
        global_train_surr = {res(), res(), res()}
    end

    train_res:update_avg(train_loss, global_train_num_samp)
    train_res_print_every:update_avg(train_loss, global_train_num_samp)

    -- exponential learning rate decay
    if i % iterations_per_epoch == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_config.learningRate = optim_config.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_config.learningRate)
        end
    end

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.5f), train_loss 1B(%dB)[allB]{surr-allB} = %6.8f (%6.8f) [%6.8f] {%2.4f,%2.4f,%2.4f}, "..
              "grad/param norm = %6.4e, time/batch = %.2fs",
              i, iterations, epoch, opt.print_every,
              train_loss, train_res_print_every.avg, train_res.avg, global_train_surr[1].avg, global_train_surr[2].avg, global_train_surr[3].avg,
              grad_params:norm() / params:norm(), time_avg.avg))
        train_res_print_every = res_avg()
        time_avg = res()
    end

    -- every now and then or on last iteration
    if i % opt.eval_val_every == 0 or i == iterations then -- or i % iterations_per_epoch == 0 then

        -- evaluate loss on validation data
        print('validating..')
        local timer = torch.Timer()
        local val_loss, perf_surrogates = evaluate('val')
        report_surrogates('val', val_perfs, val_loss, perf_surrogates, i, iterations, epoch, timer:time().real)

        val_losses[#val_losses+1] = val_loss
        val_iters[#val_iters+1] = i

        local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.val_iters = val_iters
        checkpoint.i = i
        checkpoint.epoch = epoch


        -- evaluate loss on test data (experimental only)
        print('testing (experimental)..')
        timer = torch.Timer()
        local test_loss, perf_surrogates = evaluate('test')
        report_surrogates('TEST', test_perfs, test_loss, perf_surrogates, i, iterations, epoch, timer:time().real)

        local names_table = {}
        for tmp_i = 1,#perf_surrogates do
            local test_fname = string.format('testperf_%s_epoch%.2f_surr%d.txt',opt.savefile,epoch,tmp_i)
            perf_surrogates[tmp_i]:write_json(string.format('res/%s',test_fname))
            names_table[#names_table+1] = test_fname
        end
        test_perfs_fnames[#test_perfs_fnames+1] = names_table

        checkpoint.test_perfs = test_perfs
        checkpoint.test_perfs_fnames = test_perfs_fnames
        print('saving checkpoint to ' .. savefile)
        torch.save(savefile, checkpoint)
        checkpoint_fnames[#checkpoint_fnames+1] = savefile


        -- plot the learning curve
        local xrange = torch.range(1,#checkpoint.train_losses)
        gnuplot.figure(1)
        gnuplot.pngfigure(string.format('res/curve_%s.png',opt.savefile))
        gnuplot.title('RMSprop loss over iterations')
        gnuplot.xlabel('iterations')
        gnuplot.ylabel('Loss (neg log likelihood)')
        gnuplot.plot({'train',xrange, torch.Tensor(checkpoint.train_losses), '~'},
                     {'val',torch.Tensor(checkpoint.val_iters), torch.Tensor(checkpoint.val_losses), '~'})
        gnuplot.plotflush()


        -- finishing training, find the best validation score and perform test error
        if i==iterations then
            local max_val_perf = -1 --1e10
            local best_ind = -1
            for j=1,#val_perfs do
                if val_perfs[j][1].avg>max_val_perf then -- [1] refers to first perf in perf_surrogates
                    max_val_perf = val_perfs[j][1].avg
                    best_ind = j
                end
            end
            if (#val_perfs - best_ind) < 0.1 * #val_perfs  then
                -- best validation is close to max_epochs, increase epoch by 2 more
                print('best validation too close to max_epochs, increasing max_epochs by 2 and continuing..')
                opt.max_epochs = opt.max_epochs + 2
                iterations = math.ceil(opt.max_epochs * iterations_per_epoch)
            else
                unfinished = false

                local oldfnames = checkpoint.test_perfs_fnames[best_ind]
                print(string.format('best validation acc of %6.8f was found at index %d/%d corresponding to test loss files:',
                                    max_val_perf, best_ind, #val_perfs))
                for i=1,#oldfnames do
                    local oldfname = oldfnames[i]
                    print(oldfname)
                    -- local luatab = read_json(oldfname)
                    local newfname = string.format('final-TEST-perf_%s',oldfname)
                    os.execute(string.format('cp res/%s res/%s',oldfname,newfname))
                end

                local to_write = {
                    best_val_perf = val_perfs[best_ind],
                    best_test_perf = test_perfs[best_ind],
                    opt = opt,
                    checkpoint_fname = checkpoint_fnames[best_ind]
                }
                write_json(string.format('res/final_%s.txt', opt.savefile), to_write)
            end
        end
    end

    if i % 100 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
        print([[loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.
              Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?]])
        break -- halt
    end
    if i <= loss0_min_iter then
        loss0_avg:update_avg(loss[1])
    else
        if loss0 == nil then loss0 = loss0_avg.avg end
        if loss[1] > loss0 * loss0_const then
            print(string.format('loss is exploding to %.2f at iter %d, from an initial loss0 of %.2f, aborting.', loss[1], i, loss0))
            mobdebug.start()
            local tmpa=1
            break -- halt
        end
    end
end


