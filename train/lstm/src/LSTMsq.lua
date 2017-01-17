
local LSTMsq = {}
function LSTMsq.lstm(input_size, rnn_size, num_layers, input_layer, pred_layer, dropout, drop_inputs)
  -- input_layer can be 'onehot' or 'onehotv2' (not tested yet) or 'embed' or 'raw'
  -- pred_layer can be 'softmax','softmaxtree'(under implementation),nil,'softmax_nolog','sigmoid',
  --    pred_layer can be a list of above for multiple predictions, for which each element in this list should be a table/array of elements
  --    each having type {type='softmax',output_size=10}
  -- please check the dropout implementation before using
  dropout = dropout or 0
  drop_inputs = drop_inputs or false

  -- there will be 3+2*num_layers inputs; first 3 belong to x, mask and one-minus-mask and the rest are previous cell and output activations
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  table.insert(inputs, nn.Identity()()) -- mask
  table.insert(inputs, nn.Identity()()) -- one-minus-mask
  for L = 1,num_layers do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local mask = nn.Replicate(rnn_size,2)(inputs[2])
  local one_m_mask = nn.Replicate(rnn_size,2)(inputs[3]) -- attempt to create 1-mask from mask: (nn.CSubTable()({Ones(batch_size)(inputs[2]), inputs[2]}))
  local outputs = {}
  local input_rep, input_rep_size
  for L = 1,num_layers do
    -- c,h from previous timesteps
    local prev_c = inputs[L*2+2]
    local prev_h = inputs[L*2+3]
    -- the input to this layer
    if L == 1 then
      -- drop inputs if necessary first
      x = inputs[1]
      if drop_inputs then
        if dropout > 0 then x = nn.Dropout(dropout)(x) end -- dropout on input
      end
      -- now begin the first encoding layer
      if input_layer=='onehot' then
        x = OneHot(input_size)(x)
        input_size_L = input_size
      elseif input_layer=='onehotv2' then
        x = OneHotv2(input_size)(x)
        input_size_L = input_size
      elseif input_layer=='embed' then
        x = nn.LookupTable(input_size, rnn_size)(x)
        input_size_L = rnn_size
        if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
       elseif input_layer=='raw' then
        input_size_L = input_size
      end
      -- cache the input values for skip connections
      input_rep = x
      input_rep_size = input_size_L
    else
      x = outputs[(L-1)*2]
      if dropout > 0 then x = nn.Dropout(dropout)(x) end
      x = nn.JoinTable(2)({x,input_rep})
      input_size_L = input_rep_size + rnn_size
    end
    -- evaluate the input sums at once for efficiency (annotations are experimental to identify the specific variables/nodes produced by nn.gModule)
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate({name='i2h_'..L, description = string.format('i2h at layer %d',L)})
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate({name='h2h_'..L, description = string.format('h2h at layer %d',L)})
    local all_input_sums = nn.CAddTable()({i2h, h2h}) -- output will be W_{x} * x + W_{h} * h_{t-1} as in latex

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums) -- all_input_sums is batch_size x (4*rnn_size); reshaped is batch_size x 4 xrnn_size
                                                       -- nn.Reshape() works this way when there are more elements in input than product of dimensions specified
    -- n1, n2, n3, n4 are in-gate, forget-gate, in-transform, and out-gate respectively
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4) -- nn.SplitTable(2)(reshaped) is a table of 4 elements; each element is batch_size x rnn_size

    -- peephole connections from previous cell to input and forget
    local peep_i = nn.CMul(rnn_size)(prev_c)
    local peep_f = nn.CMul(rnn_size)(prev_c)

    -- decode the gates
    local in_gate = nn.Sigmoid()( nn.CAddTable()({n1,peep_i}) )
    local forget_gate = nn.Sigmoid()( nn.CAddTable()({n2,peep_f}) )
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the cell update
    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate, in_transform})
      })
    -- if mask is 1, output will take the value of next_c, otherwise it will be prev_c, left unchanged from before
    next_c = nn.CAddTable()({
        nn.CMulTable()({mask, next_c}),
        nn.CMulTable()({one_m_mask, prev_c})
      })

    local peep_o = nn.CMul(rnn_size)(next_c) -- peephole connection from current cell to output
    local out_gate = nn.Sigmoid()( nn.CAddTable()({n3,peep_o}) )
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    next_h = nn.CAddTable()({
        nn.CMulTable()({mask, next_h}),
        nn.CMulTable()({one_m_mask, prev_h})
      })

    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  if pred_layer then
    -- set up the decoder
    local top_h
    for j=2,#outputs,2 do
      if j==2 then
        top_h = outputs[j]
      else
        top_h = nn.JoinTable(2)({top_h, outputs[j]})
      end
    end
    outputs = LSTMsq.create_pred_layer(pred_layer, rnn_size * num_layers, top_h, input_size, outputs, dropout)
  end

  return nn.gModule(inputs, outputs)
end


function LSTMsq.create_pred_layer(pred_layer, toph_size, toph, orig_input_size, outputs, dropout)
  -- local input = outputs[#outputs]
  if dropout > 0 then toph = nn.Dropout(dropout)(toph) end

  if type(pred_layer) == 'table' then
    for i=1,#pred_layer do
      local layer_type = pred_layer[i]['type']
      local output_size = pred_layer[i]['output_size']
      local proj = nn.Linear(toph_size, output_size)(toph)
      outputs = add_pred_layer(layer_type, proj, outputs)
    end
  else
    -- output_size is orig_input_size
    local proj = nn.Linear(toph_size, orig_input_size)(toph)
    outputs = add_pred_layer(pred_layer, proj, outputs)
  end
  return outputs
end

function add_pred_layer(pred_layer, proj, outputs)
  if pred_layer=='softmax_nolog' then
    local soft = nn.SoftMax()(proj)
    table.insert(outputs, soft)
  elseif pred_layer=='softmax' then
    local logsoft = nn.LogSoftMax()(proj)
    table.insert(outputs, logsoft)
  elseif pred_layer=='softmaxtree' then
    -- still under implementation
    error('softmaxtree not implemented!')
    -- local proj = nn.SoftMaxTree(rnn_size, hierarchy)(top_h)
  elseif pred_layer=='sigmoid' then
    local sig = nn.Sigmoid()(proj)
    table.insert(outputs, sig)
  else
    error('pred_layer not implemented!')
  end
  return outputs
end


function LSTMsq.h_init(rnn_size, num_layers)
  -- inputs are assumed to be zero vectors
  -- there will be 2*num_layers inputs; cell and output activations
  local inputs = {}
  for L = 1,num_layers do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  outputs = {}
  for L = 1,num_layers do
    local tmp_model = nn.Add(rnn_size)
    local c_init = tmp_model(inputs[L*2-1])
    table.insert(outputs, c_init)
    local h_init = tmp_model(inputs[L*2])
    table.insert(outputs, h_init)
    -- local h_init = nn.Identity()(c_init)
    -- table.insert(outputs, h_init)
  end

  return nn.gModule(inputs, outputs)
end


return LSTMsq
