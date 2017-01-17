local cr_data = torch.class('cr_data')

function cr_data:__init(vocab_path)
  self.vocab_path = vocab_path
  if not JSON then
    JSON = assert(loadfile 'lib/JSON.lua')()
  end

  self.map = {}
  self.imap = {}
  self.vocab_size = nil

  self:read_vocab()

end


function cr_data:read_vocab()
  local list_of_lists = util:read_json(self.vocab_path)

  self.map = {}
  self.imap = {}
  for i=1,#list_of_lists do
    self.map[list_of_lists[i]] = i
    self.imap[i] = list_of_lists[i]
  end
  self.vocab_size = #list_of_lists
end


function cr_data:encode_input(input)
  local maxt = #input
  local s1 = torch.zeros(1, self.vocab_size, maxt)
  local m1 = torch.ones(1, maxt) -- will be always a ones vector

  -- fill s1 with ones at appropriate places
  for i=1,#input do
    local term_i = input[i]
    for j=1,#term_i do
      local course_id = term_i[j]
      s1[{1,self.map[course_id],i}] = 1
    end
  end
  local k1 = self:get_k1_evaluation(s1, maxt)
  local enc_input = {s1,m1,k1}
  return enc_input
end


function cr_data:encode_output(prob, ind)
  local out = {}
  for i=1,ind:size(2) do
    local course_tup = {}
    course_tup[1] = self.imap[ind[{1,i}]]
    course_tup[2] = prob[{1,i}]
    out[#out+1] = course_tup
  end
  return out
end


function cr_data:get_k1_evaluation(s1, maxt)
  -- k1 will have encode all courses already taken
  local k1
  k1 = torch.sum(s1, 3)
  k1 = torch.squeeze(k1)
  k1[torch.gt(k1,1)] = 1 -- if same course is taken multiple times, count it as once
  k1 = k1:type('torch.LongTensor')
  return k1
end
