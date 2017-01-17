
local dm_svc = torch.class('dm_svc')

function dm_svc:__init(opt)
  -- require lib
  require 'src.util'
  require 'src.lstm_prod'
  require 'src.dm_data'
  if not JSON then
    JSON = assert(loadfile 'lib/JSON.lua')() -- note: this is global!
  end

  -- print('running server with options:',opt)
  self.opt = opt or {}
  --self.opt.config = self.opt.config or '../../train/lstm/cache/final_v2-h64-b1-lr2e-3.txt' --final_v2diae2-h64-b1-lr2e-3.txt --final_h64-b1-lr2e-3.txt
  self.opt.config = self.opt.config or '../../train/lstm/res/final_v2-h64-b1-lr2e-3.txt'
  self.models = {}
  self.datasets = {}
  self.config = nil

  self:setup()

end


function dm_svc:setup()
  self.models = {}
  self.datasets = {}

  self.config = util:read_json(self.opt.config)
  local opt = self.config['opt']

  local model_path = string.format('../../train/lstm/%s', self.config['checkpoint_fname'])
  local opt_assert = opt
  local gpuid = 3
  self.models[1] = lstm_prod(model_path, opt_assert, gpuid)
  self.datasets[1] = dm_data()
end


function dm_svc:serve(input)
  local mdl = self.models[1]
  local dataset = self.datasets[1]

  local enc_input = dataset:encode_input(input)
  local output = mdl:predict(enc_input)
  -- mobdebug.start()
  -- local tmpa=1
  local enc_output = dataset:encode_output(output)

  self:print(enc_output)
  return enc_output
end


function dm_svc:print(enc_output)
  -- print human readable representation for real-time debugging

end

