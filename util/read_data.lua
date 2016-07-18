--[[

  Functions for loading data from disk.

--]]

function HierAttnModel.read_embedding(vocab_path, emb_path)
  local vocab = HierAttnModel.Vocab(vocab_path)
  local embedding = torch.load(emb_path)
  return vocab, embedding
end

function HierAttnModel.read_queries(path,vocab,id_path, cuda)
	local queries_all = {}
	local file = io.open(path,'r')
	local line
	while true do
		line = file:read()
		if line == nil then break end
		local tokens = stringx.split(line)
		local len = #tokens
		local sent = torch.IntTensor(len)
		for i = 1, len do
			local token = tokens[i]
			sent[i] = vocab:index(token)
		end
		queries_all[#queries_all + 1] = sent
	end
	file:close()

	local query_id_map_file = io.open(id_path,'r')
	local query_id_map = {}
	local line
	while true do
		line = query_id_map_file:read()
		if line == nil then break end
		query_id = tonumber(line)
		query_id_map[#query_id_map + 1] = query_id
	end
	query_id_map_file:close()
	
	queries = {}
	for i=1,query_id_map[#query_id_map] do
		queries[i] = {}
	end

	for i=1,#query_id_map do
		l = #queries[query_id_map[i]]
        if cuda then
		    queries[query_id_map[i]][l+1] = queries_all[i]:cuda()
        else
		    queries[query_id_map[i]][l+1] = queries_all[i]
        end
	end
	return queries
end

function HierAttnModel.read_sentences(path, vocab,verbose, cuda)
	verbose = false
  local sentences = {}
  local file = io.open(path, 'r')
  local line
  while true do
    line = file:read()
	if verbose then
		print(line)
	end
    if line == nil then break end
    local tokens = stringx.split(line)
    local len = #tokens
    local sent = torch.IntTensor(len)
    for i = 1, len do
      local token = tokens[i]
      sent[i] = vocab:index(token)
    end
    if cuda then
        sentences[#sentences + 1] = sent:cuda()
    else
        sentences[#sentences + 1] = sent
    end
  end
  if verbose then
	  print('************************************************************************')
  end

  file:close()
  return sentences
end

function HierAttnModel.read_trees(parent_path)
  local parent_file = io.open(parent_path, 'r')
  local label_file
  local count = 0
  local trees = {}

  while true do
    local parents = parent_file:read()
    if parents == nil then break end
    parents = stringx.split(parents)
    for i, p in ipairs(parents) do
      parents[i] = tonumber(p)
    end

    count = count + 1
    trees[count] = HierAttnModel.read_tree(parents, nil)
	trees[count].idx = count
  end
  parent_file:close()
  return trees
end

function HierAttnModel.read_tree(parents, labels)
  local size = #parents
  local nodes = {}
  if labels == nil then labels = {} end
  local root
  for i = 1, size do
    if not nodes[i] and parents[i] ~= -1 then
      local idx = i
      local prev = nil
      while true do
        local parent = parents[idx]
        if parent == -1 then
          break
        end

        local node = HierAttnModel.TreeNode()
        if prev ~= nil then
          node:add_child(prev)
        end
        nodes[idx] = node
        node.idx = idx
        node.gold_label = labels[idx]
        if nodes[parent] ~= nil then
          nodes[parent]:add_child(node)
          break
        elseif parent == 0 then
          root = node
          root.isroot = true
          break
        else
          prev = node
          idx = parent
        end
      end
    end
  end

  -- index leaves (only meaningful for constituency trees)
  local leaf_idx = 1
  for i = 1, size do
    local node = nodes[i]
    if node ~= nil and node.num_children == 0 then
      node.leaf_idx = leaf_idx
      leaf_idx = leaf_idx + 1
    end
  end
  local tree = HierAttnModel.Tree(root, nodes)
  return tree
end


--[[

 QA

--]]
function HierAttnModel.read_toefl_dataset(dir,vocab,num_choices,prune_rate, cuda)
	local dataset = {}
	dataset.vocab = vocab

	--get to know how many sentences in one story
	dataset.num_sent = HierAttnModel.read_num_sent_table(dir .. 'num_sent',prune_rate)
	dataset.size = #dataset.num_sent

	local sent_tree_ls
	local choice_tree_ls
	local query_tree_ls

	sent_tree_ls = HierAttnModel.read_trees(string.format(dir .. 'sents_%.1f_dparents', prune_rate))
	query_tree_ls = HierAttnModel.read_trees(dir .. 'query_sep_dparents')
	choice_tree_ls = HierAttnModel.read_trees(dir .. 'choice_dparents')

	--set tree
    for _, tree in ipairs(sent_tree_ls) do
      set_spans(tree.root)
    end
    for _, tree in ipairs(query_tree_ls) do
      set_spans(tree.root)
    end
	for _,tree in ipairs(choice_tree_ls) do
		set_spans(tree.root)
	end

	--set each sentence
	local sent_ls = HierAttnModel.read_sentences(string.format(dir .. 'sents_%.1f',prune_rate) , vocab, true, cuda)
	local query_ls = HierAttnModel.read_queries(dir .. 'queries_sep', vocab,dir ..'query_id.map', cuda)
	local choice_ls = HierAttnModel.read_sentences(dir .. 'choices' , vocab, cuda)
	dataset.queries = {}
	local q_g_idx = 1
	for q=1,#query_ls do
		local one_query = {}
		for qq=1,#query_ls[q] do
			table.insert(one_query,HierAttnModel.Sentence(query_ls[q][qq],query_tree_ls[q_g_idx]))
			q_g_idx = q_g_idx + 1
		end
		table.insert(dataset.queries,one_query)
	end

	dataset.choices = {}
	local start_idx = 1
	for i=1,dataset.size do
		local one_story_choices = {}
		for c=1,num_choices do
			table.insert(one_story_choices,HierAttnModel.Sentence(choice_ls[start_idx+c-1],choice_tree_ls[start_idx+c-1]))
		end
		table.insert(dataset.choices,one_story_choices)
		start_idx = start_idx + num_choices
	end

	dataset.sents = {}
	local start_idx = 1

	for i=1,dataset.size do
		local one_story_sents = {}
		for c=1,dataset.num_sent[i] do
			table.insert(one_story_sents,HierAttnModel.Sentence(sent_ls[start_idx+c-1],sent_tree_ls[start_idx+c-1]))
		end
		table.insert(dataset.sents,one_story_sents)
		start_idx = start_idx + dataset.num_sent[i]
	end

	--set golden and answer
	dataset.golden = HierAttnModel.read_prob(dir .. 'labels', dataset.size,4, cuda)
	dataset.answers = HierAttnModel.read_answers(dir .. 'answers', dataset.size)
	dataset.container = {}
	for i=1,dataset.size do
		dataset.container[i] = {}
	end
	return dataset
end

function mysplit(inputstr,sep)
	if sep == nil then 
		sep = "%s"
	end
	local t={};i=1
	for str in string.gmatch(inputstr,"([^"..sep.."]+)") do
		t[i] = str
		i=i+1
	end
	return t
end

function HierAttnModel.read_prob(path,num_data,num_choices, cuda)
	local probs = {}
	local file = io.open(path,'r')
	for s=1,num_data do
		line = file:read()
		local tmp = mysplit(line)
		local prob = torch.Tensor(num_choices)
		for a=1,num_choices do
			prob[a] = tonumber(tmp[a])
		end
        if cuda then
		    probs[s] = prob:cuda()
        else
            probs[s] = prob
        end
	end
	file:close()
	return probs
end

function HierAttnModel.read_answers(path,num_data)
	local answers = {}
	local file = io.open(path,'r')
	for s=1,num_data do
		line = file:read()
		local tmp = mysplit(line)
		local answer = {}
		for a=1,#tmp do
			answer[a] = tonumber(tmp[a])
		end
		answers[s] = answer
	end
	file.close()
	return answers
end


function set_spans(node)
  if node.num_children == 0 then
    node.lo, node.hi = node.leaf_idx, node.leaf_idx
    return
  end

  for i = 1, node.num_children do
    set_spans(node.children[i])
  end

  node.lo, node.hi = node.children[1].lo, node.children[1].hi
  for i = 2, node.num_children do
    node.lo = math.min(node.lo, node.children[i].lo)
    node.hi = math.max(node.hi, node.children[i].hi)
  end
end

function HierAttnModel.read_labels(path, num_data)
	local labels = {}
	local file = io.open(path,'r')
	local labels = torch.IntTensor(num_data)
	for label_idx=1, num_data do
		label = file:read()
		labels[label_idx] = tonumber(label)
	end
	file:close()
	return labels:double()
end

function HierAttnModel.read_num_sent_table(path, prune_rate)
	local num_sent_table = {}
	local file = io.open(path,'r')
	local line
	while true do
		line = file:read()
		if line == nil then break end
		num_sent = tonumber(line)
		num_sent = torch.ceil(num_sent * prune_rate)
		num_sent_table[#num_sent_table + 1] = num_sent
	end
	file:close()
	return num_sent_table
end
