--[[

  Tree-LSTM testing and visualization script for QA  on the TOEFL listening comprehension

--]]

require('..')

function write_correct(pred,gold,name)
	local file = io.open(string.format("query_label/result/%s_q",name),'w')
	correct_idx = {}
	for i=1,#gold do
		for j=1,#gold[i] do
			if gold[i][j] == pred[i][j] then
				correct_idx[#correct_idx + 1] = i
				file:write(string.format("%d\n",i))
			end
		end
	end
	file:close()
	return correct_idx
end

function accuracy(pred, gold)
	local correct = 0
	for i=1,#gold do
		for j=1,#gold[i] do
			if gold[i][j] == pred[i][j] then
				correct = correct + 1
				break
			end
		end
	end
  return correct / #gold
end

-- read command line arguments
local args = lapp [[
Training script for QA on the TOEFL listening comprehension
    -m,--model  (default ham)           TreeLSTM Model architecture: [lstm, bilstm, treelstm, memn2n]
    -t,--task   (default manual)        [manual, ASR]
    -p,--prune (default 1)              Pruning rate
    -a,--path (string)                  Path to trained model
]]

printf('%-25s = %s\n', 'task',args.task)
printf('%-25s = %.2f\n', 'pruning rate',args.prune)

local model_name, model_class
if args.model == 'ham' then
    model_name = 'Hierarchical Attention Model'
    model_class = HierAttnModel.HierAttnModelToefl
elseif args.model == 'lstm' then
    model_name = 'LSTM'
    model_class = HierAttnModel.LSTMToefl
elseif args.model == 'bilstm' then
    model_name = 'Bidirectional LSTM'
    model_class = HierAttnModel.LSTMToefl
elseif args.model == 'treelstm' then
    model_name = 'TreeLSTM'
    model_class = HierAttnModel.TreeLSTMToefl
elseif args.model == 'memn2n' then
	model_name = 'End-to-end Memory Network'
	model_class = HierAttnModel.MemN2NToefl
end
header(model_name .. ' for TOEFL Question Answering')

-- directory containing dataset files
local data_dir = string.format('data/toefl/%s_trans/',args.task)

-- load vocab
local vocab = HierAttnModel.Vocab('data/toefl/manual_trans/vocab-cased.txt') -- whole vocab

-- load embeddings
print('loading word embeddings')
local emb_dir = 'data/glove/'
local emb_prefix = emb_dir .. 'glove.840B'
local emb_vocab, emb_vecs = HierAttnModel.read_embedding(emb_prefix .. '.vocab', emb_prefix .. '.300d.th')
local emb_dim = emb_vecs:size(2)

-- use only vectors in vocabulary (not necessary, but gives faster training)
local num_unk = 0
local vecs = torch.Tensor(vocab.size, emb_dim)
for i = 1, vocab.size do
	local w = string.gsub(vocab:token(i), '\\', '')  --remove escape characters
	if emb_vocab:contains(w) then
		vecs[i] = emb_vecs[emb_vocab:index(w)]
	else
		num_unk = num_unk + 1
		vecs[i]:uniform(-0.05, 0.05)
	end
end

print('unk count = ' .. num_unk)
emb_vocab = nil
emb_vecs = nil
collectgarbage()

-- load datasets
local num_choices = 4
print('loading datasets')
local test_dir = data_dir .. 'test/'
--local test_manual_dir = 'data/toefl/manual_trans/test/'
local model_save_path = args.path
print(model_save_path)
local test_dataset = HierAttnModel.read_toefl_dataset(test_dir, vocab, num_choices, args.prune)
--local test_manual_dataset = HierAttnModel.read_toefl_dataset(test_manual_dir,vocab,num_choices,args.prune)

printf('num test  = %d\n', test_dataset.size)

-- to load a saved model
local model = model_class.load(model_save_path)

-- print information
header('model configuration')
model:print_config()


header('Evaluating on test set')
local test_predictions = model:predict_dataset(test_dataset)
printf('-- test score: %.4f\n', accuracy(test_predictions, test_dataset.answers))
--write_correct(test_predictions,test_dataset.answers,args.model)
