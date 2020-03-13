import numpy as np
import pandas as pd
from sklearn import metrics
from scipy import interp
import seaborn as sn
import matplotlib.pyplot as plt
from collections import Counter
import pickle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


def savemodel(model, path):
	torch.save(model.state_dict(), path)

def clip_gradient(model, clip_value):
	params = list(filter(lambda p: p.grad is not None, model.parameters()))
	for p in params:
		p.grad.data.clamp_(-clip_value, clip_value)
	
def train_model(model, train_iter, epoch):
	total_epoch_loss = 0
	total_epoch_acc = 0
	lr=0.0001
	loss_fn = F.cross_entropy
	optim = torch.optim.Adam(model.parameters(), lr=lr)
	steps = 0
	idx = -1
	model.train()
	for inputs, labels in train_iter:
		idx += 1
		inputs = inputs.to(device)
		labels = torch.autograd.Variable(labels).long()
		labels = labels.to(device)
		optim.zero_grad()
		prediction = model(inputs)
		loss = loss_fn(prediction, labels)
		num_corrects = (torch.max(prediction, 1)[1].view(labels.size()).data == labels.data).float().sum()
		acc = num_corrects
		loss.backward()
		clip_gradient(model, 1e-1)
		optim.step()
		steps += 1
		
		if steps % 5000 == 0:
			print (f'Epoch: {epoch+1}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')
		
		total_epoch_loss += loss.item()
		total_epoch_acc += acc.item()
	path = 'drive/My Drive/dl_ass2/'+str(epoch+1)
	savemodel(model, path)    
	return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)

def eval_model(model, val_iter):
	total_epoch_loss = 0
	loss_fn = F.cross_entropy
	total_epoch_acc = 0
	model.eval()
	with torch.no_grad():
		for inputs, labels in val_iter:
			inputs = inputs.to(device)
			labels = torch.autograd.Variable(labels).long()
			labels = labels.to(device)
			prediction = model(inputs)
			loss = loss_fn(prediction, labels)
			num_corrects = (torch.max(prediction, 1)[1].view(labels.size()).data == labels.data).sum()
			acc = num_corrects
			total_epoch_loss += loss.item()
			total_epoch_acc += acc.item()
	return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)
	
def trainer(modeltype, file):
	train_loader, valid_loader = loaders(file, 'train')	
	tloss = []
	vloss = []
	tacc = []
	vacc = []
	model = choose_model(modeltype, 'train')
	for epoch in range(10):
		train_loss, train_acc = train_model(model, train_loader, epoch)
		val_loss, val_acc = eval_model(model, valid_loader)
		tloss.append(train_loss)
		vloss.append(val_loss)
		tacc.append(train_acc)
		vacc.append(val_acc)
		print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')

	with open(loc+'train_acc.pkl', 'wb') as f:
		pickle.dump(tacc, f)
	with open(loc+'val_acc.pkl', 'wb') as f:
		pickle.dump(vacc, f)
	with open(loc+'train_loss.pkl', 'wb') as f:
		pickle.dump(tloss, f)
	with open(loc+'val_loss.pkl', 'wb') as f:
		pickle.dump(vloss, f)

def predictions(modeltype, file, modelpath, name):
	model = choose_model(modeltype, 'test')
	data_iter = loaders(file, 'test')
	preds = []
	model.load_state_dict(torch.load(loc+modelpath))
	model.eval()
	with torch.no_grad():
		for inputs in data_iter:
			# print(inputs)
			inputs = inputs[0].to(device)
			# labels = torch.autograd.Variable(labels).long()
			# labels = labels.to(device)
			prediction = model(inputs)
			# num_corrects = (torch.max(prediction, 1)[1].view(labels.size()).data == labels.data).sum()
			# acc = num_corrects
			# total_epoch_acc += acc.item()
			preds.append(torch.max(prediction, 1)[1].view(1).data.detach().cpu().clone().numpy())
	# print(total_epoch_acc/len(data_iter))
	reverse_label = {}
	label_dict = {'anger': 12,'boredom': 10,'empty': 0,'enthusiasm': 2,
		'fun': 7,'happiness': 9,'hate': 8,'love': 6,'neutral': 3,
		'relief': 11,'sadness': 1,'surprise': 5,'worry': 4}
	for i in range(13):
		for key in label_dict:
			if label_dict[key] == i:
				reverse_label[i] = key
	ID = []
	sentiment = []
	y_pred = []
	for p in preds:
		y_pred.append(p[0])
	for i, lab in enumerate(y_pred):
		ID.append(i+1)
		sentiment.append(reverse_label[lab])
	df = pd.DataFrame(list(zip(ID, sentiment)), 
				columns =['ID', 'Class'])
	df.to_csv(loc+name+'.csv', encoding='utf-8', index=False)

if __name__=="__main__":
	# train_file = 'train_data.csv'
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	modeltype = 'rnn' #rnn/lstm
	modelpath = 'rnn/4' #lstm/1
	# trainer(modeltype, train_file)
	test_file = 'test.csv'
	predictions(modeltype, test_file, modelpath, 'results')
