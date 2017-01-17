
local LSTMsq = require 'src.LSTMsq'

local pred = {}
function pred.create_layer(input_size, pred_layer, dropout)
  -- pred_layer can be 'softmax','softmaxtree'(under implementation),nil,'softmax_nolog','sigmoid',
  --    pred_layer can be a list of above for multiple predictions, for which each element in this list should be a table of type {type='softmax',output_size=10}
  dropout = dropout or 0

  -- there will be 3+2*num_layers inputs; first 3 belong to x, mask and one-minus-mask and the rest are previous cell and output activations
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x


  local input = inputs[1]
  local outputs = {}
  LSTMsq.crate_pred_layer(pred_layer, input_size, input, outputs, dropout)

  return nn.gModule(inputs, outputs)
end

return pred
