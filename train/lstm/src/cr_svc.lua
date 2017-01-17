
local cr_svc = torch.class('cr_svc')

function cr_svc:__init(opt)
  -- require lib
  require 'src.util'
  require 'src.lstm_prod'
  require 'src.cr_data'
  if not JSON then
    JSON = assert(loadfile 'lib/JSON.lua')()
  end

  print('running server with options:',opt)
  self.opt = opt or {}
  self.opt.config = self.opt.config or 'cache/bestmodels_setlstmv2.json'
  self.models = {}
  self.datasets = {}
  self.config = nil

  self:setup()

end


function cr_svc:setup()

  self.config = util:read_json(self.opt.config)

  self.models = {}
  self.datasets = {}
  for term_k=1,7 do
    local term_k_str = string.format('%s',term_k)
    local opt = self.config[term_k_str]
    assert(opt['term_k']==term_k_str)

    local model_path = string.format('cache/%s', opt['model_fname'])
    local data_path = string.format('cache/voc-%s.txt', term_k_str)
    local opt_assert = {
      reconstruct_wt = opt['reconstruct_wt'],
      rnn_size = opt['rnn_size'],
      term_k = opt['term_k']
    }
    local gpuid = term_k % 4 -- 4 gpus in system
    self.models[term_k] = lstm_prod(model_path, opt_assert, gpuid)
    self.datasets[term_k] = cr_data(data_path)
  end

end


function cr_svc:serve(input)
  local term_k = #input
  local mdl = self.models[term_k]
  local dataset = self.datasets[term_k]

  local enc_input = dataset:encode_input(input)
  local prob, ind = mdl:predict(enc_input)

  local enc_output = dataset:encode_output(prob, ind)

  return enc_output
end


function cr_svc:print(enc_output)
  for i=1,#enc_output do

    for j=1,#a do
    end
  end

end

