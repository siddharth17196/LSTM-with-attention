'''
USAGE - 

1. CALL PLOTTER - (RETURNS ACCURACY AND LOSS PLOTS)

2. CALL CREATE_CONFUSION - 
	(RETURNS ROC CURVES AND CONFUSION MATRICES)
'''


def plots(item_train, item_valid, title):
	'''
	PLotting the loss and accuracy plots
	'''
	plt.figure()
	plt.plot(item_train, label='training')
	plt.plot(item_valid, label='validation')
	plt.title(title)
	plt.legend(loc='best')
	plt.savefig(loc+title+'.png')

def plotter(locs):	
	'''
	Helper function for plotting the 
	1. loss plots
	2. accuracy plots
	'''
	with open(loc+'train_acc.pkl', 'rb') as f:
		train_acc = pickle.load(f)
	with open(loc+'val_acc.pkl', 'rb') as f:
		val_acc = pickle.load(f)
	with open(loc+'train_loss.pkl', 'rb') as f:
		train_loss = pickle.load(f)
	with open(loc+'val_loss.pkl', 'rb') as f:
		val_loss = pickle.load(f)

	plts = [train_acc, val_acc, train_loss, val_loss]
	title = ['accuracy RNN', 'loss RNN']
	plots(plts[0], plts[1], title[0])
	plots(plts[2], plts[3], title[1])

def plot_confusion(array, title):       
	'''
	PLotting the confusion matrix
	'''
	df_cm = pd.DataFrame(array, range(13), range(13))
	plt.figure(figsize=(20,20))
	sn.heatmap(df_cm, annot=True, annot_kws={"size": 12})
	plt.savefig(loc+title+'.png')
	plt.show()

def plot_roc(y_test, y_pred,name):
	'''
	PLotting the roc curves for the predictions
	'''
	y_test = []
	y_pred = []
	for t, p in zip(labs,preds):
		y_test.append(t[0])
		y_pred.append(p[0])
	classes = list(set(y_test))
	print(len(classes))
	print(metrics.accuracy_score(y_test, y_pred))
	y_test = label_binarize(y_test,classes)
	y_pred = label_binarize(y_pred,classes)
	plt.figure()
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	n_classes = len(classes)

	mean_fpr = np.linspace(0,1,100)
	tprs = []
	aucs = []
	for i in range(n_classes):
		fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])
		aucs.append(roc_auc[i])
		tprs.append(interp(mean_fpr, fpr[i], tpr[i]))
		tprs[-1][0] = 0.0
		plt.plot(fpr[i], tpr[i], lw=1, alpha=0.3)

	mean_tpr = np.mean(tprs, axis=0)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	std_auc = np.std(aucs)
	plt.plot(mean_fpr, mean_tpr, color='b',
			label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
			lw=2, alpha=.8)

	std_tpr = np.std(tprs, axis=0)
	tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
	tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
	plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
					label=r'$\pm$ 1 std. dev.')

	plt.plot([0, 1], [0, 1], color='navy', alpha=0.8, lw=2, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title(name)
	plt.legend(loc="lower right")
	plt.savefig(loc+name+'roc'+'.png')
 
def create_confusion(model, data_iter,name):    
	'''
	Helper function for creating
	1. Confusion Matrix
	2. ROC plots
	'''
	preds = []
	labs = []
	total_epoch_acc = 0
	model.eval()
	with torch.no_grad():
		for inputs, labels in data_iter:
			inputs = inputs.to(device)
			labels = torch.autograd.Variable(labels).long()
			labels = labels.to(device)
			prediction = model(inputs)
			labs.append(labels.data.detach().cpu().clone().numpy())
			# print(type(labs[0]))
			# break
			num_corrects = (torch.max(prediction, 1)[1].view(labels.size()).data == labels.data).sum()
			acc = num_corrects
			total_epoch_acc += acc.item()
			preds.append(torch.max(prediction, 1)[1].view(labels.size()).data.detach().cpu().clone().numpy())
	cm = confusion_matrix(labs, preds)
	print(total_epoch_acc/len(data_iter))
	plot_roc(labs, preds, name)
	plot_confusion(cm,name)
