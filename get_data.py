'''
PRE PROCESSES THE DATA 
1. REMOVES PUNCTUATIONS
2. REMOVES HTMLS, TAGS
3. PAD TO 30 

MAKE PYTORCH DATALOADERS 
80:20 TRAIN/VALID SPLIT FOR TRAINING AND 
USE THE WHOLE DATA IN CASE OF TEST

USAGE - CALL LOADERS (RETURNS THE DATA-LOADERS)
'''


def pad_features(content_ints, seq_length):
	features = np.zeros((len(content_ints), seq_length), dtype=int)
	for i, row in enumerate(content_ints):
		features[i, -len(row):] = np.array(row)[:seq_length]
	return features

def preprocess(name, func):
	dat = pd.read_csv(loc+name)
	dat.drop(dat.columns[[0, 1,2]], axis = 1, inplace = True) 
	dat = dat.tail(100)
	content = np.array(dat['content'])
	if func == 'train':
		labels = np.array(dat['sentiment'])
	punctuation = '!"#$%&\'()*+,-./:;<=>?[\\]^_`{|}~'
	# get rid of punctuation
	all_content = 'separator'.join(content)
	all_content = all_content.lower()
	all_text = ''.join([c for c in all_content if c not in punctuation])

	# split by new lines and spaces
	content_split = all_text.split('separator')
	all_text = ' '.join(content_split)

	# create a list of words
	words = all_text.split()

	# get rid of web address, twitter id, and digit
	new_content = []
	for cont in content_split:
		cont = cont.split()
		new_text = []
		for word in cont:
			if (word[0] != '@') & ('http' not in word) & (~word.isdigit()):
				# if word not in stop_words:
				new_text.append(word)
		new_content.append(new_text)

	counts = Counter(words)
	vocab = sorted(counts, key=counts.get, reverse=True)
	vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
	print(vocab_to_int)
	## use the dict to tokenize each review in reviews_split
	## store the tokenized reviews in content_ints
	content_ints = []
	for cont in new_content:
		content_ints.append([vocab_to_int[word] for word in cont])

	# stats about vocabulary
	# print('Unique words: ', len((vocab_to_int))) 
	# print()

	# # print tokens in first review
	# print('Tokenized review: \n', content_ints[:1])
	# c_label = Counter(labels)

	# %matplotlib inline
	content_len = [len(x) for x in content_ints]
	pd.Series(content_len).hist()
	plt.show()
	pd.Series(content_len).describe()
	if func == 'train':
		label_dict = {'anger': 12,'boredom': 10,'empty': 0,'enthusiasm': 2,
		'fun': 7,'happiness': 9,'hate': 8,'love': 6,'neutral': 3,
		'relief': 11,'sadness': 1,'surprise': 5,'worry': 4}

		encoded_labels = []
		for sentiment in labels:
			encoded_labels.append(label_dict[sentiment])
		encoded_labels = np.asarray(encoded_labels)

		content_ints = np.asarray([content_ints[i] for i, l in enumerate(content_len) if l>0])
		encoded_labels = np.asarray([encoded_labels[i] for i, l in enumerate(content_len) if l>0])

		seq_length = 30
		features = pad_features(content_ints, seq_length=seq_length)

		## test statements 
		assert len(features)==len(content_ints), "The features should have as many rows as reviews."
		assert len(features[0])==seq_length, "Each feature row should contain seq_length values."

		return features, encoded_labels
	
	else:
		content_ints = np.asarray([content_ints[i] for i, l in enumerate(content_len) if l>0])
		seq_length = 30
		features = pad_features(content_ints, seq_length=seq_length)
		assert len(features)==len(content_ints), "The features should have as many rows as reviews."
		assert len(features[0])==seq_length, "Each feature row should contain seq_length values."
		return features

def loaders(file, func):	
	if func=='train':
		features, encoded_labels = preprocess(file,func)
		split_frac = 0.80
		## split data into training, validation, and test data (features and labels, x and y)
		split_idx = int(len(features)*split_frac)
		train_x, val_x = features[:split_idx], features[split_idx:]
		train_y, val_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

		# create Tensor datasets
		train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
		valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))

		# dataloaders
		train_loader = DataLoader(train_data, shuffle=True)#, batch_size=batch_size, drop_last=True)
		valid_loader = DataLoader(valid_data, shuffle=True)#, batch_size=batch_size, drop_last=True
		return train_loader, valid_loader
	else:
		features = preprocess(file, func)
		test_x = features
		test_data = TensorDataset(torch.from_numpy(test_x))
		test_loader = DataLoader(test_data)
		print(test_loader)
		
		return test_loader