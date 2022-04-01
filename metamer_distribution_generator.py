"""
Code to generate metamer task distributions from a given abstraction (supply name with command line argument). This will save a .pt file in data/ that the enviornments will interface with. 
""" 
import torch
import torch.nn.functional as F
import numpy as np   
import sys 
from abstract_task_distributions import *   
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class Net(torch.nn.Module):
	def __init__(self,S=7):
		super(Net, self).__init__()
		self.fc1 = torch.nn.Linear(S*S, S*S)
		self.fc2 = torch.nn.Linear(S*S, S*S)
		self.fc3 = torch.nn.Linear(S*S, S*S)

	def forward(self,x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x=self.fc3(x) 

		return F.sigmoid(x)

S=7 

network=Net() 
#network.load_state_dict(torch.load("data/perceptrons_task_distribution_"+name+".pt"))   
#network.eval()

def gibbs_sample(S=7,numSweeps=20,network=network):
	
	M=np.random.choice([0,1],size=S*S,replace=True)
	idxs=np.arange(S*S)
	for sweep in range(numSweeps):
		np.random.shuffle(idxs)
		for i in idxs:
			M_eval=M.copy()
			M_eval[i]=-1
			masked=torch.from_numpy(M_eval).float()
			preds=network(masked).detach().numpy().reshape((S*S,))
			if np.random.rand()<preds[i]:
				M[i]=1
			else:
				M[i]=0
	return M 
	

def batch_gibbs(S=7,numSweeps=20,batch_size=430,network=network):
	M=torch.from_numpy(np.random.choice([0,1],size=S*S*batch_size,replace=True).reshape((batch_size,S*S))).float()
	idxs=np.arange(S*S)
	for sweep in range(numSweeps):
		np.random.shuffle(idxs)
		for i in idxs:
			M_eval=M.clone().detach()
			M_eval[:,i]=-1
			preds=network(M_eval)[:,i]
			r=torch.rand(batch_size)
			M[r<preds,i]=1
			M[r>=preds,i]=0
	return M.detach().numpy() 


if __name__=='__main__':
	name=sys.argv[1]
	if '4x4' in name:
		S=4
	else:
		S=7 
	net=Net(S=S)
	lr=0.0002
	#heldout=set([tuple(x) for x in np.load('held_out/all.npy')])
	#configurations=np.load('data/gsp_unstructured_4x4.npy') 
	#configuration_probs=np.load('data/gsp_unstructured_4x4_probs.npy')
	#configurations=np.load('data/shape_task_distribution.npy').reshape((-1,49))


	
	optimizer=torch.optim.Adam(net.parameters())
	criterion = torch.nn.BCELoss()

	#true=configurations.copy()  
	accuracy_buffer=[]
	for epoch in range(4000):
		#true=batch_gibbs() 
		#np.random.shuffle(true)

		#sample_idxs=np.random.choice(np.arange(configurations.shape[0]),size=340,replace=False,p=configuration_probs)
		#true=configurations[sample_idxs]
		true=np.asarray([sample_task(name) for _ in range(400)]).reshape((-1,S*S))



		#num_changes=np.random.choice(list(range(1,10)),size=1)[0]
		num_changes=1
		change_idxs=np.zeros(true.shape).astype('bool')
		for i in range(true.shape[0]):
			idxs=np.random.choice(np.arange(true.shape[1]),size=num_changes,replace=False)
			change_idxs[i,idxs]=True


		data=true.copy()
		data[change_idxs]=-1
		masked=torch.from_numpy(data).float()

		labels=torch.from_numpy(true).float()
		tensor_idxs=torch.from_numpy(change_idxs).bool() 

		preds=net(masked)

		loss=criterion(preds[tensor_idxs],labels[tensor_idxs])
		y_hat=(preds[tensor_idxs]>0.5).int().numpy()
		y=labels[tensor_idxs].int().numpy()


		print(epoch,loss,np.sum(y_hat==y)/y_hat.shape[0])
		
		net.zero_grad()
		loss.backward()
		optimizer.step()
		if len(accuracy_buffer)<5:
			accuracy_buffer.append(np.sum(y_hat==y)/y_hat.shape[0])
		else:
			if np.mean(accuracy_buffer)>=0.99: 
				break
			else:
				accuracy_buffer=[]
	if name=='gsp_4x4':
		torch.save(net.state_dict(),"data/gsp_4x4_fc_generator.pt")
	elif name=='grammar_4x4':
		torch.save(net.state_dict(),"data/grammar_4x4_fc_generator.pt")
	else:
		torch.save(net.state_dict(),"data/perceptrons_task_distribution_"+name+".pt")
	

	net.eval()  
	

	#samples=batch_gibbs(network=net,S=7)  
	#np.save('data/gsp_4x4_null.npy',samples)
	
	#network.load_state_dict(torch.load("data/perceptrons_task_distribution_"+name+".pt"))  
	sample=batch_gibbs(S=S,batch_size=625,network=net).reshape((-1,S,S)) 
	check=np.sum(np.sum(sample,axis=(1,2))<3) 
	if '4x4' in name: 
		np.save('data/'+name+"_large_sample.npy",sample) 
	else:	
		np.save('data/perceptrons_7x7_'+name+"_large_sample.npy",sample)
	"""
	starts=[]
	for i in range(sample.shape[0]):
		choices=np.vstack(np.where(sample[i]>0))
		choice=np.random.choice(np.arange(choices.shape[1]))
		starts.append(choices[:,choice])
	starts=np.asarray(starts) 
	np.save('data/perceptrons_7x7_'+name+"_large_sample_starts.npy",starts) 
	"""  