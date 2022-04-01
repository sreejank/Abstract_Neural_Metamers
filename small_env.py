"""
Training enviornment for meta-rl agent for the abstract task distributions (first set of experiments)
"""
import gym
from gym.utils import seeding
from PIL import Image as PILImage
from gym.spaces import Box
from gym.spaces import Discrete
import numpy as np
from itertools import product  
from itertools import permutations 
import pickle 
from grid_grammar import * 
from abstract_task_distributions import * 
from metamer_distribution_generator import * 
"""
buff_idx=[1000] 
buffer=[[]]
def gibbs_sample_buffer(S=7,numSweeps=20,network=network):
	if buff_idx[0]>=1000:
		buffer[0]=batch_gibbs(S=S,numSweeps=numSweeps,network=network,batch_size=1000)
		buff_idx[0]=0
	M=buffer[0][buff_idx[0]]
	buff_idx[0]+=1
	return M  
"""



class BattleshipEnv(gym.Env):  
	
	reward_range = (-float('inf'), float('inf'))
	metadata = {'render.modes': ['human', 'rgb_array'],'video.frames_per_second' : 3}
	
	def __init__(self,rules='chain',n_board=7,hold_out=0,permute=0,pretrain=0,render_img=0):
		
		self.viewer = None
		self.seed()
		action_converter=[]
		for i in range(n_board):
			for j in range(n_board): 
				action_converter.append((i,j))
		self.action_converter=np.asarray(action_converter)
		self.n_board=n_board

		self.hold_out=hold_out
		self.rules=rules
		self.permute=permute
		self.pretrain=pretrain 
		self.render_img=render_img
		if self.permute:
			self.permutation=np.load('7x7_permutation.npy')

		if hold_out==-1:
			self.heldout=np.load('data/'+self.rules+'_sample.npy').reshape((-1,7,7))
			if self.permute:
				for i in range(len(self.heldout)):
					self.heldout[i]=self.heldout[i][self.permutation]
			self.maze_idx=0 
			self.maze=np.reshape(self.heldout[self.maze_idx],(7,7))
			start=np.load('data/'+self.rules+'_sample_starts.npy')[self.maze_idx]

		else:
			if hold_out>0:
				#heldout=[tuple(generate_grid(rules,n=self.n_board).flatten()) for _ in range(hold_out)]
				heldout=np.load('data/'+self.rules+'_sample.npy').reshape((-1,49))
				if self.permute:
					for i in range(len(heldout)):
						heldout[i]=heldout[i][self.permutation]
				self.heldout=set([tuple(x) for x in heldout]) 
			
			if 'null' not in self.rules and 'gsp' in self.rules:
				self.total_boards=np.load('data/'+self.rules+"_task_distribution.npy").reshape((-1,7,7))
				self.total_boards_starts=np.load('data/'+self.rules+"_task_distribution_starts.npy")
				r_idx=np.random.choice(np.arange(self.total_boards.shape[0]))
				gen=(self.total_boards[r_idx],self.total_boards_starts[r_idx])
				grid=gen[0]
			elif 'null' not in self.rules and 'gsp' not in self.rules:
				gen=sample_task(self.rules)
				grid=gen[0]

			else:
				if 'gsp' in self.rules:
					network.load_state_dict(torch.load('data/'+self.rules+"_generator.pt"))
				else:
					network.load_state_dict(torch.load("data/perceptrons_task_distribution_"+self.rules[:-5]+".pt"))
				
				self.size_buffer=1000 
				self.gibbs_buffer=batch_gibbs(S=7,numSweeps=20,batch_size=self.size_buffer,network=network).reshape((-1,7,7))
				self.gibbs_idx=0

				if self.gibbs_idx>=self.size_buffer:
					self.gibbs_buffer=batch_gibbs(S=7,numSweeps=20,batch_size=self.size_buffer,network=network).reshape((-1,7,7))
					self.gibbs_idx=0
				grid=self.gibbs_buffer[self.gibbs_idx]
				self.gibbs_idx=self.gibbs_idx+1 

				while np.sum(grid)<3 or np.sum(grid)>=40: 
					if self.gibbs_idx>=self.size_buffer:
						self.gibbs_buffer=batch_gibbs(S=7,numSweeps=20,batch_size=self.size_buffer,network=network).reshape((-1,7,7))
						self.gibbs_idx=0
					grid=self.gibbs_buffer[self.gibbs_idx]
					self.gibbs_idx=self.gibbs_idx+1 

				gen=grid
			

			if self.permute:
				grid=np.reshape(grid.flatten()[self.permutation],(7,7))
			if hold_out>0:
				while tuple(grid.flatten()) in self.heldout:
					if 'null' not in self.rules and 'gsp' in self.rules:
						self.total_boards=np.load('data/'+self.rules+"_task_distribution.npy").reshape((-1,7,7))
						self.total_boards_starts=np.load('data/'+self.rules+"_task_distribution_starts.npy")
						r_idx=np.random.choice(np.arange(self.total_boards.shape[0]))
						gen=(self.total_boards[r_idx],self.total_boards_starts[r_idx])
						grid=gen[0]
					elif 'null' not in self.rules and 'gsp' not in self.rules:
						gen=sample_task(self.rules)
						grid=gen[0]
					else:
						if self.gibbs_idx>=self.size_buffer:
							self.gibbs_buffer=batch_gibbs(S=7,numSweeps=20,batch_size=self.size_buffer,network=network).reshape((-1,7,7))
							self.gibbs_idx=0
						grid=self.gibbs_buffer[self.gibbs_idx]
						self.gibbs_idx=self.gibbs_idx+1 
						while np.sum(grid)<3 or np.sum(grid)>=40:
							if self.gibbs_idx>=self.size_buffer:
								self.gibbs_buffer=batch_gibbs(S=7,numSweeps=20,batch_size=self.size_buffer,network=network).reshape((-1,7,7))
								self.gibbs_idx=0
							grid=self.gibbs_buffer[self.gibbs_idx]
							self.gibbs_idx=self.gibbs_idx+1 
						gen=grid
				if len(gen)==2:
					grid,start=gen 
				else:
					grid=gen 
					hit_idx=np.where(grid==1)
					choice=np.random.choice(list(range(len(hit_idx[0]))),size=1)
					start=(hit_idx[0][choice],hit_idx[1][choice])
				if self.permute:
					grid=np.reshape(grid.flatten()[self.permutation],(7,7))
			self.maze=grid

		self.board=np.ones(self.maze.shape)*-1
		self.current_position=start 
		self.board[self.current_position[0],self.current_position[1]]=1
		self.num_hits=0
		self.self_hits={}
		if self.render_img:
			self.observation_space = Box(low=-1, high=1, shape=(7*7*1+n_board*n_board+1,), dtype=np.int8)
		else:
			self.observation_space = Box(low=-1, high=1, shape=(n_board*n_board+n_board*n_board+1,), dtype=np.int8)
		self.action_space = Discrete(np.prod(self.maze.shape))
		self.nA=n_board*n_board

		self.prev_reward=0
		self.prev_action=np.zeros((self.nA,))

		self.valid_actions=[1 for _ in range(self.nA)]
		
		
	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
		
	def step(self, action):
		#print(self.tot_solves)
		#print(action,self.valid_actions,self.valid_actions[action]) 
		prev_position=self.current_position
		self.current_position=self.action_converter[action]
		reward=0
		
		if self.board[self.current_position[0],self.current_position[1]]==-1:
			if self.maze[self.current_position[0],self.current_position[1]]==1:
				self.board[self.current_position[0],self.current_position[1]]=1
				self.num_hits+=1
				reward=1
			else:
				self.board[self.current_position[0],self.current_position[1]]=0
				reward=-1
		else:
			reward=-2
			if (self.current_position[0],self.current_position[1]) not in self.self_hits.keys():
				self.self_hits[(self.current_position[0],self.current_position[1])]=1
			else:
				self.self_hits[(self.current_position[0],self.current_position[1])]+=1
			
			
						
			
		if self._is_goal():
			reward=+10
			done = True
			if self.hold_out==-1:
				self.maze_idx+=1
		else:
			done = False

		p_action=self.prev_action
		p_reward=self.prev_reward
		self.prev_action=np.zeros((self.nA,))
		self.prev_action[action]=1
		self.prev_reward=reward 

		if self.pretrain:
			board_tensor=torch.from_numpy(self.board.flatten()).float()
			obs=network(board_tensor).detach().numpy()
		else:
			if self.render_img:
				obs=self.get_image().flatten()
			else:
				obs=self.board.flatten()

		obs_array=np.concatenate((obs,p_action,[p_reward]))

		self.valid_actions=[0 for _ in range(self.nA)]
		for i in range(self.nA):
			pos=self.action_converter[i]
			if self.board[pos[0],pos[1]]==0:
				self.valid_actions[i]=1
		
		
		return obs_array, reward, done, {'valid_actions':self.valid_actions}
	
	def get_action_mask(self):
		return self.valid_actions
	def _is_goal(self):
		return np.sum(self.board==1)==np.sum(self.maze==1)
	
	def get_image(self):
		"""
		img=np.empty((28,28, 1), dtype=np.int8) 
		#fills=[[0,0,255],[255,0,0],[255,255,255]]
		#fills=[0,128,255]
		for r in range(self.board.shape[0]):
			for c in range(self.board.shape[1]):
				#fill=fills[self.board[r,c].astype('int')]
				img[4*r:4*r+4,4*c:4*c+4]=self.board[r,c]
		return img  
		"""
		return np.reshape(self.board,(7,7,1))
		
	
	def set_task(self,task,start):
		self.maze = task
		self.board=np.ones(self.maze.shape)*-1
		self.current_position=start 

		self.num_hits=0
		self.self_hits={}
		self.board[self.current_position[0],self.current_position[1]]=1

		obs=self.board.flatten()
		obs_array=np.concatenate((obs,self.prev_action,[self.prev_reward]))
		
		return obs_array 
	
	def reset(self):
		if self.hold_out==-1:
			self.maze=np.reshape(self.heldout[self.maze_idx%len(self.heldout)],(7,7))
			start=np.load('data/'+self.rules+'_sample_starts.npy')[self.maze_idx%len(self.heldout)] 
		else:
			if 'null' not in self.rules and 'gsp' in self.rules:
				#self.total_boards=np.load('data/'+self.rules+"_task_distribution.npy").reshape((-1,7,7))
				#self.total_boards_starts=np.load('data/'+self.rules+"_task_distribution_starts.npy")
				r_idx=np.random.choice(np.arange(self.total_boards.shape[0]))
				gen=(self.total_boards[r_idx],self.total_boards_starts[r_idx])
				grid=gen[0]
			elif 'null' not in self.rules and 'gsp' not in self.rules:
				gen=sample_task(self.rules)
			else:
				if self.gibbs_idx>=self.size_buffer:
					self.gibbs_buffer=batch_gibbs(S=7,numSweeps=20,batch_size=self.size_buffer,network=network).reshape((-1,7,7))
					self.gibbs_idx=0
				grid=self.gibbs_buffer[self.gibbs_idx]
				self.gibbs_idx=self.gibbs_idx+1
				while np.sum(grid)<3 or np.sum(grid)>=40:
					if self.gibbs_idx>=self.size_buffer:
						self.gibbs_buffer=batch_gibbs(S=7,numSweeps=20,batch_size=self.size_buffer,network=network).reshape((-1,7,7))
						self.gibbs_idx=0
					grid=self.gibbs_buffer[self.gibbs_idx]
					self.gibbs_idx=self.gibbs_idx+1

				gen=grid
			if len(gen)==2:
				grid,start=gen 
				grid=gen[0]
			else:
				grid=gen 
				hit_idx=np.where(grid==1)
				choice=np.random.choice(list(range(len(hit_idx[0]))),size=1)
				start=(hit_idx[0][choice],hit_idx[1][choice])
			if self.permute:
				grid=np.reshape(grid.flatten()[self.permutation],(7,7))

			if self.hold_out>0:
				while tuple(grid.flatten()) in self.heldout:
					if 'null' not in self.rules and 'gsp' in self.rules:
						#self.total_boards=np.load('data/'+self.rules+"_task_distribution.npy").reshape((-1,7,7))
						#self.total_boards_starts=np.load('data/'+self.rules+"_task_distribution_starts.npy")
						r_idx=np.random.choice(np.arange(self.total_boards.shape[0]))
						gen=(self.total_boards[r_idx],self.total_boards_starts[r_idx])
						grid=gen[0]
					elif 'null' not in self.rules and 'gsp' not in self.rules:
						gen=sample_task(self.rules)
						grid=gen[0]
					else:
						if self.gibbs_idx>=self.size_buffer:
							self.gibbs_buffer=batch_gibbs(S=7,numSweeps=20,batch_size=self.size_buffer,network=network).reshape((-1,7,7))
							self.gibbs_idx=0
						grid=self.gibbs_buffer[self.gibbs_idx] 
						self.gibbs_idx=self.gibbs_idx+1
						while np.sum(grid)<3 or np.sum(grid)>=40:
							if self.gibbs_idx>=self.size_buffer:
								self.gibbs_buffer=batch_gibbs(S=7,numSweeps=20,batch_size=self.size_buffer,network=network).reshape((-1,7,7))
								self.gibbs_idx=0
							grid=self.gibbs_buffer[self.gibbs_idx]
							self.gibbs_idx=self.gibbs_idx+1
						gen=grid 
					if len(gen)==2: 
						grid,start=gen 
					else:
						grid=gen 
						hit_idx=np.where(grid==1)
						choice=np.random.choice(list(range(len(hit_idx[0]))),size=1)
						start=(hit_idx[0][choice],hit_idx[1][choice])
					if self.permute:
						grid=np.reshape(grid.flatten()[self.permutation],(7,7))
			self.maze=grid

		self.board=np.ones(self.maze.shape)*-1
		self.current_position=start 
		self.board[self.current_position[0],self.current_position[1]]=1

		self.num_hits=0
		self.self_hits={}
		if self.pretrain:
			board_tensor=torch.from_numpy(self.board.flatten()).float()
			obs=network(board_tensor).detach().numpy()
		else:
			if self.render_img:
				obs=self.get_image().flatten()
			else:
				obs=self.board.flatten()

		obs_array=np.concatenate((obs,self.prev_action,[self.prev_reward]))
		self.valid_actions=[1 for _ in range(self.nA)] 
		return obs_array
	
	def render(self, mode='human', max_width=500): 
		img = self.get_image()
		img = np.asarray(img).astype(np.uint8)
		img_height, img_width = img.shape[:2]
		ratio = max_width/img_width
		img = PILImage.fromarray(img).resize([int(ratio*img_width), int(ratio*img_height)])
		img = np.asarray(img)
		if mode == 'rgb_array':
			return img
		elif mode == 'human':
			from gym.envs.classic_control.rendering import SimpleImageViewer
			if self.viewer is None:
				self.viewer = SimpleImageViewer()
			self.viewer.imshow(img)
			
			return self.viewer.isopen
	def close(self): 
		if self.viewer is not None:
			self.viewer.close()
			self.viewer = None
	
def register_small_env(env_id,rules,n_board=7,hold_out=0,permute=0,max_episode_steps=60,pretrain=0,render_img=0):  
	gym.envs.register(id=env_id, entry_point=BattleshipEnv, max_episode_steps=max_episode_steps,kwargs={'rules':rules,'n_board':n_board,'hold_out':hold_out,'permute':permute,'pretrain':pretrain,'render_img':render_img})  
