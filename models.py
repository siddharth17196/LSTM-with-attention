'''
DEFINATION OF MODELS FOR 
1. LSTM
2. SIMPLE RNN

USGAE  - CALL CHOOSE_MODEL (RETURNS THE CHOSEN MODEL ARCHITECTURE)
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMAttentionModel(nn.Module):
	def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, n_layers, drop_prob=0.5):
		super(LSTMAttentionModel, self).__init__()
		self.batch_size = batch_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.embedding_length = embedding_length
		self.n_layers = n_layers

		self.embedding = nn.Embedding(vocab_size, embedding_length)
		self.lstm = nn.LSTM(embedding_length, hidden_size, n_layers, 
							dropout=drop_prob, batch_first = True,)
		self.midl = nn.Linear(hidden_size*3, 150)
		self.drop = nn.Dropout(p=0.3)
		self.label = nn.Linear(150, output_size)
		
	def attention_net(self, lstm_output, final_state):
		hidden = final_state
		hidden = hidden.squeeze(1)
		hidden = torch.t(hidden)
		# print(hidden.squeeze(0).unsqueeze(2).shape)
		# attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
		# soft_attn_weights = F.softmax(attn_weights, 1)
		# new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
		attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(0)).squeeze(2)
		soft_attn_weights = F.softmax(attn_weights, 1)
		new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights).squeeze(2)
		new_hidden_state = torch.flatten(new_hidden_state)
		return new_hidden_state.unsqueeze(0)
	
	def forward(self, x, batch_size=1):
		x = x.long()
		embeds = self.embedding(x)
		
		h_0 = Variable(torch.zeros(self.n_layers, self.batch_size, self.hidden_size).to(device))
		c_0 = Variable(torch.zeros(self.n_layers, self.batch_size, self.hidden_size).to(device))
			
		output, (final_hidden_state, final_cell_state) = self.lstm(embeds, (h_0, c_0))
		attn_output = self.attention_net(output, final_hidden_state)
		m = self.drop(self.midl(attn_output))
		logits = self.label(m)

		return logits


class RNNAttentionModel(nn.Module):
	def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, n_layers, drop_prob=0.5):
		super(RNNAttentionModel, self).__init__()
		self.batch_size = batch_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.embedding_length = embedding_length
		self.n_layers = n_layers

		self.embedding = nn.Embedding(vocab_size, embedding_length)
		self.lstm = nn.RNN(embedding_length, hidden_size, n_layers, 
							dropout=drop_prob, batch_first = True,)
		self.midl = nn.Linear(hidden_size*3, 150)
		self.drop = nn.Dropout(p=0.3)
		self.label = nn.Linear(150, output_size)
		
	def attention_net(self, lstm_output, final_state):
		hidden = final_state
		hidden = hidden.squeeze(1)
		hidden = torch.t(hidden)
		# print(hidden.squeeze(0).unsqueeze(2).shape)
		# attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
		# soft_attn_weights = F.softmax(attn_weights, 1)
		# new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

		attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(0)).squeeze(2)
		soft_attn_weights = F.softmax(attn_weights, 1)
		new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights).squeeze(2)
		new_hidden_state = torch.flatten(new_hidden_state)
		return new_hidden_state.unsqueeze(0)
	
	def forward(self, x, batch_size=1):
		x = x.long()
		embeds = self.embedding(x)
		
		h_0 = Variable(torch.zeros(self.n_layers, self.batch_size, self.hidden_size).to(device))
			
		output, final_hidden_state = self.lstm(embeds, h_0)
		attn_output = self.attention_net(output, final_hidden_state)
		m = self.drop(self.midl(attn_output))
		logits = self.label(m)

		return logits

def choose_model(modeltype, func):	
	if func=='train':
		val = len(vocab_to_int)
	else:
		val = 48591

	vocab_size = val+1 # 48591+1  +1 for the 0 padding + our word tokens
	output_size = 13
	embedding_length = 300
	hidden_size = 256
	n_layers = 3

	if modeltype == 'lstm':
		model = LSTMAttentionModel(vocab_size = vocab_size,
						   output_size = output_size, 
						   embedding_length = embedding_length,
						   hidden_size = hidden_size,
						   n_layers = n_layers,
						   batch_size = 1).to(device)
	else:	
		model = RNNAttentionModel(vocab_size = vocab_size,
						   output_size = output_size, 
						   embedding_length = embedding_length,
						   hidden_size = hidden_size,
						   n_layers = n_layers,
						   batch_size = 1).to(device)
	return model