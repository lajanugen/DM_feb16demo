-- A class to handle incremental computation of confidence intervals (of mean)
-- for scenarios with random sampling when making estimates of mean
--
-- ebar computed is the standard error (http://en.wikipedia.org/wiki/Standard_error)
-- In other words, ebar is standard deviation (sigma) divided by sqrt(num_samples)
-- So, for 95% confidence intervals, you will have to use a width of 1.96*ebar
-- Upper 95% limit = avg + 1.96*ebar
-- Lower 95% limit = avg - 1.96*ebar

local res = torch.class('res')

function res:__init()
  self.num_samp = 0
  self.avg = 0
  self.tmp_var = 0
  self.ebar = -10
  self.conf_intr = -10.0 -- one-sided confidence interval, ie, 95% confidence interval is avg +/- conf_intr
end

function res:update_avg(curr_sample)
  self.num_samp = self.num_samp + 1
  local prev_avg = self.avg
  self.avg = self.avg + (curr_sample - self.avg) / self.num_samp
  self.tmp_var = self.tmp_var + (curr_sample - prev_avg) * (curr_sample - self.avg)
end

function res:update_ebar()
  if self.num_samp>1 then
    self.ebar = math.sqrt( self.tmp_var / (self.num_samp * (self.num_samp-1)) )
    self.conf_intr = 1.96*self.ebar
  end
end

function res:write_json(fname)
  local luatable = {num_samp = self.num_samp, avg = self.avg, ebar = self.ebar, conf_intr = self.conf_intr}
  write_json(fname, luatable)
end


local res_avg = torch.class('res_avg')

function res_avg:__init()
  self.num_samp = 0
  self.avg = 0
end

function res_avg:update_avg(curr_avg, num_samp)
  if num_samp==0 then
    return
  end
  local prev_avg = self.avg
  local prev_num_samp = self.num_samp
  self.num_samp = prev_num_samp + num_samp
  self.avg = (prev_avg*prev_num_samp + curr_avg*num_samp) / self.num_samp
end

