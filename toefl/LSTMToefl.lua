--[[

  LSTM-Reader for Question Answering on TOEFL test (4 choices)

--]]

local LSTMToefl = torch.class('HierAttnModel.LSTMToefl')

function LSTMToefl:__init(config)
	self.num_choices    = 4
	self.mem_dim        = config.mem_dim           or 75 --LSTM memory dimension
	self.lr             = config.lr                or 0.05
	self.emb_lr         = config.emb_lr            or 0.1
	self.batch_size     = config.batch_size        or 32
	self.reg            = config.reg               or 1e-4
	self.mlp_units      = config.mlp_units         or 128
	self.dropout        = config.dropout           or 0.0
    self.structure      = config.structure         or 'lstm' -- {lstm, bilstm}
    self.num_layers     = config.num_layers        or 1

    self.cuda           = config.cuda


    -- word embedding
    self.emb_dim = config.emb_vecs:size(2)
    if self.cuda then
        self.emb = nn.LookupTable(config.emb_vecs:size(1), self.emb_dim):cuda()
        self.in_zeros = torch.zeros(self.emb_dim):cuda()
        self.emb.weight:copy(config.emb_vecs:cuda())
    else
        self.emb = nn.LookupTable(config.emb_vecs:size(1), self.emb_dim)
        self.in_zeros = torch.zeros(self.emb_dim)
        self.emb.weight:copy(config.emb_vecs)
    end


    -- optimizer configuration
    self.optim_state = { learningRate = self.lr }

    -- negative log likelihood optimization objective
    if self.cuda then
	    self.criterion = nn.DistKLDivCriterion():cuda()
    else
	    self.criterion = nn.DistKLDivCriterion()
    end
	local lstm_config = {
		in_dim = self.emb_dim,
		mem_dim = self.mem_dim,
        num_layers = self.num_layers,
        gate_output = true,
        cuda = self.cuda
	}

    self.lstm_choices = {}
    if self.structure == 'lstm' then
        self.lstm = HierAttnModel.LSTM(lstm_config)
        for c=1, 4 do
            self.lstm_choices[c] = HierAttnModel.LSTM(lstm_config)
        end
    elseif self.structure == 'bilstm' then
        self.lstm_choices_b = {}
        self.lstm = HierAttnModel.LSTM(lstm_config)
        self.lstm_b = HierAttnModel.LSTM(lstm_config)
        for c=1, 4 do
            self.lstm_choices[c] = HierAttnModel.LSTM(lstm_config)
            self.lstm_choices_b[c] = HierAttnModel.LSTM(lstm_config)
        end
    else
        error('invalid LSTM type: ' .. self.structure)
    end

	self.sim_output_module = self:new_sim_module()

	local modules = nn.Parallel() -- to get the param in different module
		:add(self.lstm)
		:add(sim_output_module)
	self.params, self.grad_params = modules:getParameters()

    for c=1, 4 do
        share_params(self.lstm_choices[c], self.lstm)
    end

    if self.structure == 'bilstm' then
        share_params(self.lstm_b, self.lstm)
        for c=1, 4 do
            share_params(self.lstm_choices_b[c], self.lstm)
        end
    end
end

function LSTMToefl:new_sim_module()
	local choice_inputs = {}
	local choice_inputs_b = {}
	local choice_vecs = {}
	local choice_vecs_b = {}
	local similarities = {}
	local mem_out
	local inputs_sim

	local mem_out_in = nn.Identity()()
	if self.num_layers == 1 then
		mem_out = mem_out_in
	else
		mem_out = nn.JoinTable(1)(mem_out_in)
	end
	
	if self.structure == 'lstm' then
		for i=1, self.num_choices do
			choice_inputs[i] = nn.Identity()()

			if self.num_layers == 1 then
				choice_vecs[i] = choice_inputs[i]
			else
				choice_vecs[i] = nn.JoinTable(1)(choice_inputs[i])
			end
			similarities[i] = nn.CosineDistance(){mem_out,choice_vecs[i]}
		end
		inputs_sim = {mem_out_in,choice_inputs[1],choice_inputs[2],choice_inputs[3],choice_inputs[4]}
	else
		local mem_out_in_b = nn.Identity()()
		if self.num_layers == 1 then
			mem_out_b = mem_out_in_b
		else
			mem_out_b = nn.JoinTable(1)(mem_out_in_b)
		end
		for i=1, self.num_choices do
			choice_inputs[i] = nn.Identity()()
			choice_inputs_b[i] = nn.Identity()()
			if self.num_layers == 1 then
				choice_vecs[i] = choice_inputs[i]
				choice_vecs_b[i] = choice_inputs_b[i]
			else
				choice_vecs[i] = nn.JoinTable(1)(choice_inputs[i])
				choice_vecs_b[i] = nn.JoinTable(1)(choice_inputs_b[i])
			end
			similarities[i] = nn.CosineDistance(){nn.JoinTable(1){mem_out,mem_out_b},nn.JoinTable(1){choice_vecs[i],choice_vecs_b[i]}}
		end
		inputs_sim = {mem_out_in,mem_out_in_b,choice_inputs[1],choice_inputs_b[1],choice_inputs[2],choice_inputs_b[2],choice_inputs[3],choice_inputs_b[3],choice_inputs[4],choice_inputs_b[4]}
	end
	local sim_all = nn.JoinTable(1){similarities[1],similarities[2],similarities[3],similarities[4]}
	local sim_module = nn.gModule(inputs_sim,{sim_all})

	local sim_output_module = nn.Sequential()
		:add(sim_module)
		:add(nn.LogSoftMax())

    if self.cuda then
        sim_output_module = sim_output_module:cuda()
    end
	
	return sim_output_module
end

function LSTMToefl:train(dataset)

	--revoke sub module
	self.lstm:training()
    if self.structure == 'bilstm' then
        self.lstm_b:training()
    end
    for c=1, 4 do
        self.lstm_choices[c]:training()
        if self.structure == 'bilstm' then
            self.lstm_choices_b[c]:training()
        end
    end

	local indices = torch.randperm(dataset.size) --shuffle

    for i = 1, dataset.size, self.batch_size do
        xlua.progress(i, dataset.size)
        local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

        local feval = function(x)
            self.grad_params:zero()
            self.emb:zeroGradParameters()

            local loss = 0
            for j = 1, batch_size do
                local idx = indices[i + j - 1]

				--choices
				local choice_vecs_f = {}
				local choice_vecs_b = {}
				local emb_choices = {}
				for c=1,self.num_choices do
					self.emb:forward(dataset.choices[idx][c].sent)
					local emb_choice
                    if self.cuda then
                        emb_choice = torch.CudaTensor(self.emb.output:size()):copy(self.emb.output)
                    else
                        emb_choice = torch.Tensor(self.emb.output:size()):copy(self.emb.output)
                    end
					table.insert(emb_choices,emb_choice)
                    local choice_vec_f
					local choice_vec_b
                    if self.structure == 'lstm' then
					    choice_vec_f = self.lstm_choices[c]:forward(emb_choices[c])
                    elseif self.structure == 'bilstm' then
						choice_vec_f = self.lstm_choices[c]:forward(emb_choices[c])
						choice_vec_b = self.lstm_choices_b[c]:forward(emb_choices[c],true)
						table.insert(choice_vecs_b,choice_vec_b)
                    end
					table.insert(choice_vecs_f,choice_vec_f)
				end

                local story = dataset.sents[idx][1].sent
				for s=2,dataset.num_sent[idx] do
                    story = torch.cat(story, dataset.sents[idx][s].sent)
                end

				for q=1,#dataset.queries[idx] do
                    story = torch.cat(story, dataset.queries[idx][q].sent)
                end

				self.emb:forward(story)
				local emb_story
                if self.cuda then
                    emb_story = torch.CudaTensor(self.emb.output:size()):copy(self.emb.output)
                else
                    emb_story = torch.Tensor(self.emb.output:size()):copy(self.emb.output)
                end
                local rep_f,rep_b,inputs_sim
                if self.structure == 'lstm' then
                    rep_f = self.lstm:forward(emb_story)
					pred = self.sim_output_module:forward{rep_f,choice_vecs_f[1],choice_vecs_f[2],choice_vecs_f[3],choice_vecs_f[4]}
                elseif self.structure == 'bilstm' then
					rep_f = self.lstm:forward(emb_story)
                    rep_b = self.lstm_b:forward(emb_story, true)
					pred = self.sim_output_module:forward{rep_f,rep_b,choice_vecs_f[1],choice_vecs_b[1],choice_vecs_f[2],choice_vecs_b[2],choice_vecs_f[3],choice_vecs_b[3],choice_vecs_f[4],choice_vecs_b[4]}
                end
				
				loss = loss + self.criterion:forward(pred,dataset.golden[idx])
				local sim_grad = self.criterion:backward(pred,dataset.golden[idx])
				local lstm_grad
				for c=1,self.num_choices do
                    if self.structure == 'lstm' then
						lstm_grad = self.sim_output_module:backward({rep_f,choice_vecs_f[1],choice_vecs_f[2],choice_vecs_f[3],choice_vecs_f[4]}, sim_grad)
					    self:LSTM_backward(self.lstm_choices[c], dataset.choices[idx][c].sent, emb_choices[c], lstm_grad[c+1])
                    elseif self.structure == 'bilstm' then
						lstm_grad = self.sim_output_module:backward({rep_f,rep_b,choice_vecs_f[1],choice_vecs_b[1],choice_vecs_f[2],choice_vecs_b[2],choice_vecs_f[3],choice_vecs_b[3],choice_vecs_f[4],choice_vecs_b[4]}, sim_grad)
					    self:BiLSTM_backward(self.lstm_choices[c], self.lstm_choices_b[c], dataset.choices[idx][c].sent, emb_choices[c], lstm_grad[2*c+1],lstm_grad[2*c+2])
                    end
				end

                -- backward thru story
                if self.structure == 'lstm' then
                    self:LSTM_backward(self.lstm, story, emb_story, lstm_grad[1])
                elseif self.structure == 'bilstm' then
                    self:BiLSTM_backward(self.lstm, self.lstm_b, story, emb_story, lstm_grad[1],lstm_grad[2])
                end

            end

            loss = loss / batch_size
            self.grad_params:div(batch_size)
            self.emb.gradWeight:div(batch_size)
			self.emb:updateParameters(self.emb_lr) -- norm to num input

            -- regularization
            loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
            self.grad_params:add(self.reg, self.params)
            return loss, self.grad_params
        end

        optim.rmsprop(feval, self.params, self.optim_state)
    end
    xlua.progress(dataset.size, dataset.size)
end

-- LSTM backward propagation
function LSTMToefl:LSTM_backward(lstm, sent, inputs, rep_grad)
    local grad
    if self.num_layers == 1 then
        if self.cuda then
            grad = torch.zeros(sent:nElement(), self.mem_dim):cuda()
        else
            grad = torch.zeros(sent:nElement(), self.mem_dim)
        end
        grad[sent:nElement()] = rep_grad
    else
        if self.cuda then
            grad = torch.zeros(sent:nElement(), self.num_layers, self.mem_dim):cuda()
        else
            grad = torch.zeros(sent:nElement(), self.num_layers, self.mem_dim)
        end
        for l = 1, self.num_layers do
            grad[{sent:nElement(), l, {}}] = rep_grad[l]
        end
    end
    local input_grads = lstm:backward(inputs, grad)
    return input_grads
end

-- Bidirectional LSTM backward propagation
function LSTMToefl:BiLSTM_backward(lstm, lstm_b, sent, inputs, rep_grad,rep_b_grad)
    local grad, grad_b
    if self.num_layers == 1 then
        if self.cuda then
            grad   = torch.zeros(sent:nElement(), self.mem_dim):cuda()
            grad_b = torch.zeros(sent:nElement(), self.mem_dim):cuda()
        else
            grad   = torch.zeros(sent:nElement(), self.mem_dim)
            grad_b = torch.zeros(sent:nElement(), self.mem_dim)
        end
        grad[sent:nElement()] = rep_grad
        grad_b[1] = rep_b_grad
    else
        if self.cuda then
            grad   = torch.zeros(sent:nElement(), self.num_layers, self.mem_dim):cuda()
            grad_b = torch.zeros(sent:nElement(), self.num_layers, self.mem_dim):cuda()
        else
            grad   = torch.zeros(sent:nElement(), self.num_layers, self.mem_dim)
            grad_b = torch.zeros(sent:nElement(), self.num_layers, self.mem_dim)
        end
        for l = 1, self.num_layers do
            grad[{sent:nElement(), l, {}}] = rep_grad[l]
            grad_b[{1, l, {}}] = rep_b_grad[l]
        end
    end
    local input_grads = lstm:backward(inputs, grad)
    local input_grads_b = lstm_b:backward(inputs, grad_b, true)
    return input_grads + input_grads_b
end

function LSTMToefl:predict(sents, choices, queries, num_sent, num_answers)
	
	--revoke sub module
    self.lstm:evaluate()
    if self.structure == 'bilstm' then
        self.lstm_b:evaluate()
    end
    for c=1, 4 do
        self.lstm_choices[c]:evaluate()
        if self.structure == 'bilstm' then
            self.lstm_choices_b[c]:evaluate()
        end
    end

    local prediction = {}

    --choices
    local choice_vecs = {}
	local choice_vecs_b = {}
    local emb_choices = {}
    for c=1,self.num_choices do
        self.emb:forward(choices[c].sent)
        local emb_choice
        if self.cuda then
            emb_choice = torch.CudaTensor(self.emb.output:size()):copy(self.emb.output)
        else
            emb_choice = torch.Tensor(self.emb.output:size()):copy(self.emb.output)
        end
        table.insert(emb_choices,emb_choice)
        local choice_vec,choice_vec_b
        if self.structure == 'lstm' then
            choice_vec = self.lstm_choices[c]:forward(emb_choices[c])
        elseif self.structure == 'bilstm' then
			choice_vec = self.lstm_choices[c]:forward(emb_choices[c])
			choice_vec_b = self.lstm_choices[c]:forward(emb_choices[c],true)
			table.insert(choice_vecs_b,choice_vec_b)
        end
        table.insert(choice_vecs,choice_vec)
    end

    local story = sents[1].sent
    for s=2,num_sent do
        story = torch.cat(story, sents[s].sent)
    end

    --query
    for q=1,#queries do
        story = torch.cat(story, queries[q].sent)
    end

    self.emb:forward(story)
    local emb_story
    if self.cuda then
        emb_story = torch.CudaTensor(self.emb.output:size()):copy(self.emb.output)
    else
        emb_story = torch.Tensor(self.emb.output:size()):copy(self.emb.output)
    end
    local rep,rep_b
    if self.structure == 'lstm' then
        rep = self.lstm:forward(emb_story)
		pred = self.sim_output_module:forward{rep,choice_vecs[1], choice_vecs[2], choice_vecs[3], choice_vecs[4]}
    elseif self.structure == 'bilstm' then
		rep = self.lstm:forward(emb_story)
		rep_b = self.lstm:forward(emb_story,true)
		pred = self.sim_output_module:forward{rep,rep_b,choice_vecs[1], choice_vecs_b[1],choice_vecs[2],choice_vecs_b[2] ,choice_vecs[3], choice_vecs_b[3],choice_vecs[4],choice_vecs_b[4]}
    end


	prediction = argmax_n(pred,num_answers)

    self.lstm:forget()
    if self.structure == 'bilstm' then
        self.lstm_b:forget()
    end
    for c=1, 4 do
        self.lstm_choices[c]:forget()
        if self.structure == 'bilstm' then
            self.lstm_choices_b[c]:forget()
        end
    end

    return prediction
end

function LSTMToefl:predict_dataset(dataset)
    local predictions = {}
    for i = 1, dataset.size do
        xlua.progress(i, dataset.size)
        table.insert(predictions, self:predict(dataset.sents[i], dataset.choices[i], dataset.queries[i], dataset.num_sent[i], #dataset.answers[i]))
    end
    return predictions
end

function argmax_n(v,n)
	sorted,indices = torch.sort(v,true)
	ret , _ = torch.sort(indices[{{1,n}}])
	return ret
end

function argmax(v)
    local idx = 1
    local max = v[1]
    for i = 2, v:size(1) do
        if v[i] > max then
            max = v[i]
            idx = i
        end
    end
    return idx
end

function LSTMToefl:print_config()
    local num_params = self.params:size(1)
    printf('%-25s = %d\n',   'num params', num_params)
	printf('%-25s = %d\n',   'Number of choices', self.num_choices)
    printf('%-25s = %s\n',   'LSTM structure', self.structure)
    printf('%-25s = %d\n',   'LSTM layers', self.num_layers)
    printf('%-25s = %d\n',   'LSTM memory dim', self.mem_dim)
    printf('%-25s = %.2e\n', 'learning rate', self.lr)
    printf('%-25s = %.2e\n', 'word vector learning rate', self.emb_lr)
    printf('%-25s = %d\n',   'minibatch size', self.batch_size)
    printf('%-25s = %.2e\n', 'regularization strength', self.reg)
    printf('%-25s = %d\n',   'word vector dim', self.emb_dim)
end

function LSTMToefl:save(path)
    local config = {
		num_choices             = self.num_choices,
        mem_dim                 = self.mem_dim,
		lr                      = self.lr,
		emb_lr                  = self.emb_lr,
		batch_size              = self.batch_size,
		reg                     = self.reg,
		structure               = self.structure,
        num_layers              = self.num_layers,
        emb_vecs                = self.emb.weight:float(),
        cuda                    = self.cuda,
    }

    torch.save(path, {
        params = self.params,
        config = config,
    })
end

function LSTMToefl.load(path)
    local state = torch.load(path)
    local model = HierAttnModel.LSTMToefl.new(state.config)
    model.params:copy(state.params)
    return model
end
