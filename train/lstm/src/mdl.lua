
local mdl = torch.class('mdl')

function mdl:__init(opt)
  self.opt = opt

  self.params = nil
  self.grad_params = nil

  self:require()
  self:setup_model() -- needs change

  self:ship_model_gpu()
  self:flatten_params()
  self:make_clones()
end


function mdl:require()
end

