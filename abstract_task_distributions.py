"""
Code to generate all the abstract task distributions. 
"""
import numpy as np 
import pickle
import itertools 
from grid_grammar import *
def diag1symmetry(x):
    return x[0,1]==x[1,2] and x[0,0]==x[2,2] and x[1,0]==x[2,1]
def diag2symmetry(x):
    return np.sum(x.T==x)==9
def horizontal_symmetry(x):
    return np.sum(x[0,:]==x[2,:])==3
def vertical_symmetry(x):
    return np.sum(x[:,0]==x[:,2])==3

lst=[]
for b in list(itertools.product([0,1],repeat=9)):
    board=np.reshape(b,(3,3))
    axes=[]
    
    if diag1symmetry(board) or diag2symmetry(board):
        axes.append('d1')
    if diag2symmetry(board):
        axes.append('d2')
    if horizontal_symmetry(board):
        axes.append('h')
    if vertical_symmetry(board):
        axes.append('v')
    if len(axes)>0 and np.sum(board)>=3:
        #lst.append((board,axes))
        lst.append((board,['d1','d2','h','v']))

def translate(stamp,axes,options=-1,axis='random'):
    if axis=='random':
        axis=np.random.choice(axes)
    board=np.zeros((7,7))
    board[2:5,2:5]=stamp 
    if options==-1:
        options=np.random.choice([0,1,2])
    if axis=='h':
        if options==0:
            board[0:3,2:5]+=stamp
            board=(board>0).astype('int')
            if np.sum(board[0:3,2:5]==board[2:5,2:5])<9:
                return []
        elif options==1:
            board[4:7,2:5]+=stamp
            board=(board>0).astype('int') 
            if np.sum(board[4:7,2:5]==board[2:5,2:5])<9:
                return [] 
        elif options==2:
            board[0:3,2:5]+=stamp
            board[4:7,2:5]+=stamp
            board=(board>0).astype('int')
            if np.sum(board[0:3,2:5]==board[2:5,2:5])<9 or np.sum(board[4:7,2:5]==board[2:5,2:5])<9:
                return []  

    elif axis=='v':
        if options==0:
            board[2:5,0:3]+=stamp
            board=(board>0).astype('int')
            if np.sum(board[2:5,0:3]==board[2:5,2:5])<9:
                return []
        elif options==1:
            board[2:5,4:7]+=stamp 
            board=(board>0).astype('int')
            if np.sum(board[2:5,4:7]==board[2:5,2:5])<9:
                return []
        elif options==2:
            board[2:5,0:3]+=stamp
            board[2:5,4:7]+=stamp 
            board=(board>0).astype('int')
            if np.sum(board[2:5,4:7]==board[2:5,2:5])<9 or np.sum(board[2:5,0:3]==board[2:5,2:5])<9:
                return []
    elif axis=='d1':
        if options==0:
            board[0:3,0:3]+=stamp
            board=(board>0).astype('int')
            if np.sum(board[0:3,0:3]==board[2:5,2:5])<9:
                return []
        elif options==1:
            board[4:7,4:7]+=stamp 
            board=(board>0).astype('int')
            if np.sum(board[4:7,4:7]==board[2:5,2:5])<9:
                return []
        elif options==2:
            board[0:3,0:3]+=stamp
            board[4:7,4:7]+=stamp 
            board=(board>0).astype('int')
            if np.sum(board[4:7,4:7]==board[2:5,2:5])<9 or np.sum(board[0:3,0:3]==board[2:5,2:5])<9:
                return []
    elif axis=='d2':
        if options==0:
            board[0:3,4:7]+=stamp
            board=(board>0).astype('int')
            if np.sum(board[0:3,4:7]==board[2:5,2:5])<9:
                return []
        elif options==1:
            board[4:7,0:3]+=stamp 
            board=(board>0).astype('int')
            if np.sum(board[4:7,0:3]==board[2:5,2:5])<9:
                return []
        elif options==2:
            board[0:3,4:7]+=stamp
            board[4:7,0:3]+=stamp 
            board=(board>0).astype('int')
            if np.sum(board[4:7,0:3]==board[2:5,2:5])<9 or np.sum(board[0:3,4:7]==board[2:5,2:5])<9:
                return []
    return board



def copy_task_distribution_sample():
    loc=np.zeros((7,7))
    kernel=np.random.choice([0,1],size=9).reshape((3,3))
    while kernel.sum()==0:
        kernel=np.random.choice([0,1],size=9).reshape((3,3))
    center1=np.random.choice([1,2,3,4,5],size=2)
    
    loc[center1[0]-1:center1[0]+2,center1[1]-1:center1[1]+2]=1 
    
    options=[]
    for i in range(1,6):
        for j in range(1,6):
            loc2=np.zeros((7,7))
            center2=np.asarray([i,j]).astype('int')
            loc2[center2[0],center2[1]]=1
            for addX in [-2,-1,0,1,2]:
                for addY in [-2,-1,0,1,2]:
                    if center2[0]+addX>=0 and center2[0]+addX<7 and center2[1]+addY>=0 and center2[1]+addY<7:
                        loc2[center2[0]+addX,center2[1]+addY]=1 

            #print(center2,loc2)
            if np.sum(loc*loc2)==0:
                options.append(center2)
    #print(center1,len(options))
    while len(options)==0:
        loc=np.zeros((7,7))
        kernel=np.random.choice([0,1],size=9).reshape((3,3))
        while kernel.sum()==0:
            kernel=np.random.choice([0,1],size=9).reshape((3,3))
        center1=np.random.choice([1,2,3,4,5],size=2)
        
        loc[center1[0]-1:center1[0]+2,center1[1]-1:center1[1]+2]=1 
        
        options=[]
        for i in range(1,6):
            for j in range(1,6):
                loc2=np.zeros((7,7))
                center2=np.asarray([i,j]).astype('int')
                loc2[center2[0],center2[1]]=1
                for addX in [-2,-1,0,1,2]:
                    for addY in [-2,-1,0,1,2]:
                        if center2[0]+addX>=0 and center2[0]+addX<7 and center2[1]+addY>=0 and center2[1]+addY<7:
                            loc2[center2[0]+addX,center2[1]+addY]=1 

                #print(center2,loc2)
                if np.sum(loc*loc2)==0:
                    options.append(center2)
    center2=options[np.random.choice(np.arange(len(options)))]
    


    g=np.zeros((7,7))
    g[center1[0]-1:center1[0]+2,center1[1]-1:center1[1]+2]=kernel 
    g[center2[0]-1:center2[0]+2,center2[1]-1:center2[1]+2]=kernel
    return g 

def generate_single_symmetry(horizontal=None,start=None):
    axis_symmetry=np.random.choice(np.arange(1,6))
    g=np.zeros((7,7))
    horizontal=np.random.choice([0,1])
    if horizontal:
        start=[np.random.choice(np.arange(0,7)),axis_symmetry]
    else:
        start=[axis_symmetry,np.random.choice(np.arange(0,7))]
    #print(start,axis_symmetry,horizontal)
    g[start[0],start[1]]=1

    for i in range(4):
        reds=np.vstack(np.where(g==1)).T 
        if horizontal:
            choices=np.asarray([coord for coord in reds if coord[1]<=axis_symmetry])
        else:
            choices=np.asarray([coord for coord in reds if coord[0]<=axis_symmetry])
        neighbors=[]
        if horizontal==0:
            addXs=[-1,0]
            addYs=[-1]
        else:
            addXs=[-1]
            addYs=[-1,0]
        for choice in choices: 
            for addX in addXs:
                    for addY in addYs:
                        if choice[0]+addX>=0 and choice[0]+addX<7 and choice[1]+addY>=0 and choice[1]+addY<7:
                            if g[choice[0]+addX,choice[1]+addY]==0:
                                if horizontal:
                                    distance=np.abs(choice[1]+addY-axis_symmetry)
                                    if axis_symmetry+distance<7:
                                        neighbors.append((choice[0]+addX,choice[1]+addY))
                                else:
                                    distance=np.abs(choice[0]+addX-axis_symmetry)
                                    if axis_symmetry+distance<7:
                                        neighbors.append((choice[0]+addX,choice[1]+addY))

        if len(neighbors)==0:
            return g
        chosen=neighbors[np.random.choice(np.arange(len(neighbors)))]
        x,y=chosen 
        g[x,y]=1
        if horizontal:
            #print(x,y)
            distance=np.abs(y-axis_symmetry)
            g[x,axis_symmetry+distance]=1 
            #print((x,y),distance,(x,axis_symmetry+distance))
        else:
            distance=np.abs(x-axis_symmetry)
            g[axis_symmetry+distance,y]=1
            #print((x,y),distance,(axis_symmetry+distance,y))
    return g 
    

    

def symmetric_task_distribution_sample():
    b=generate_single_symmetry()
    while np.sum(b)>=15 or np.sum(b)<=2:
        b=generate_single_symmetry()
    return b 


def connected_task_distribution_sample():
    b=np.zeros((7,7))
    start=np.random.choice([2,3,4,5],size=2,replace=True)
    b[start[0],start[1]]=-1
    iters=np.random.choice([1,2,3])
    for iter in range(iters):
        ons=np.vstack(np.where(b<0)).T 
        choices=[]
        for on_idx in ons:
            x,y=on_idx
            if x-1>=1 and b[x-1,y]==0:
                choices.append((x-1,y))
            if y-1>=1 and b[x,y-1]==0:
                choices.append((x,y-1))
            if x+1<6 and b[x+1,y]==0:
                choices.append((x+1,y))
            if y+1<6 and b[x,y+1]==0:
                choices.append((x,y+1))
        for choice in choices:
            if np.random.choice([0,1]):
                b[choice[0],choice[1]]=-1 

    ons=np.vstack(np.where(b<0)).T 
    choices=[]
    for on_idx in ons:
        x,y=on_idx
        if x-1>=0 and b[x-1,y]==0:
            choices.append((x-1,y))
        if y-1>=0 and b[x,y-1]==0:
            choices.append((x,y-1))
        if x+1<7 and b[x+1,y]==0:
            choices.append((x+1,y))
        if y+1<7 and b[x,y+1]==0:
            choices.append((x,y+1))

        if x-1>=0 and y-1>=0 and b[x-1,y-1]==0:
            choices.append((x-1,y-1))
        if x-1>=0 and y+1<7 and b[x-1,y+1]==0:
            choices.append((x-1,y+1))
        if x+1<7 and y-1>=0 and b[x+1,y-1]==0:
            choices.append((x+1,y-1))
        if x+1<7 and y+1<7 and b[x+1,y+1]==0:
            choices.append((x+1,y+1))
    for choice in choices:
        b[choice[0],choice[1]]=1
    b[np.where(b<0)]=0
    return b 

def rectangle_task_distribution_sample():
    corner1=np.random.choice(np.arange(7),size=2,replace=True)
    corner2=np.random.choice(np.arange(7),size=2,replace=True)
    while corner1[0]==corner2[0] or corner1[1]==corner2[1]:
        corner1=np.random.choice(np.arange(7),size=2,replace=True)
        corner2=np.random.choice(np.arange(7),size=2,replace=True)
    b=np.zeros((7,7))
    b[corner1[0],corner1[1]]=1
    b[corner2[0],corner2[1]]=1
    x=min(corner1[0],corner2[0])
    y=max(corner1[0],corner2[0])+1
    for i in range(x,y):
        b[i,corner1[1]]=1
        b[i,corner2[1]]=1
    x=min(corner1[1],corner2[1])
    y=max(corner1[1],corner2[1])+1
    for j in range(x,y):
        b[corner1[0],j]=1
        b[corner2[0],j]=1
    return b 



def tree_task_distribution_sample():
    grid=generate_grid('tree')[0]
    while np.sum(grid)<=3:
        grid=generate_grid('tree')[0]
    return grid 

def pyramid_task_distribution_sample():
    base_center=[np.random.choice([1,2,3,4,5,6]),np.random.choice([1,2,3,4,5])]
    sizes=[3]
    if base_center[0]>=2 and base_center[1]>=2 and base_center[1]<=4:
        sizes.append(5)
    if base_center[0]>=3 and base_center[1]==3:
        sizes.append(7)
    size=np.random.choice(sizes)
    g=np.zeros((7,7))
    g[base_center[0],base_center[1]]=1
    g[base_center[0],base_center[1]-1]=1
    g[base_center[0],base_center[1]+ 1]=1
    leftmost=base_center[1]-1
    rightmost=base_center[1]+1
    if size>=5:
        g[base_center[0],base_center[1]-2]=1
        g[base_center[0],base_center[1]+2]=1
        leftmost=base_center[1]-2
        rightmost=base_center[1]+2
    if size>=7:
        g[base_center[0],base_center[1]-3]=1
        g[base_center[0],base_center[1]+3]=1
        leftmost=base_center[1]-3
        rightmost=base_center[1]+3
    #print(base_center,leftmost,rightmost)
    #plt.imshow(g)
    #plt.figure()
    left=[base_center[0],leftmost]
    right=[base_center[0],rightmost]
    while left[1]!=right[1]:
        #print(left,right)
        left[0]-=1
        right[0]-=1
        left[1]+=1
        right[1]-=1
        g[left[0],left[1]:right[1]+1]=1
        #g[right[0],right[1]]=1
        #print(left,right)
    return np.rot90(g,k=np.random.choice([1,2,3,4]))


def zigzag_task_distribution_sample():
    x0=np.random.choice([0,1,2,3,4,5])
    y0=np.random.choice([0,1,2,3,4,5])
    step=np.random.choice(list(range(1,7-max(x0,y0)))) 
    x=x0 
    y=y0
    b=np.zeros((7,7)) 
    #print(x0,y0,step)
    while x+step<=6 and y+step<=6:
        b[x:x+step+1,y]=1
        b[x+step,y:y+step+1]=1
        x=x+step 
        y=y+step 
    return np.rot90(b,k=np.random.choice([1,2,3,4]))

def cross_task_distribution_sample():
    if np.random.choice([0,1]):
        g=np.zeros((7,7))
        start=[np.random.choice([1,2,3,4,5]),np.random.choice([0,1,2,3,4,5,6])]
        h=start[0]
        a=start[1]
        options=[x for x in [0,1,2,3,4,5,6] if abs(x-a)>1]
        b=np.random.choice(options)
        x=min(a,b)
        y=max(a,b)
        g[h,x:y+1]=1
        
        v=np.random.choice(list(range(x+1,y)))
        xx=np.random.choice(list(range(0,h)))
        yy=np.random.choice(list(range(h+1,7))) 
        g[xx:yy+1,v]=1 
        return g
    else:
    
        g=np.zeros((7,7))
        start=[np.random.choice([0,1,2,3,4]),np.random.choice([0,1,2,3,4])]
        max_size=7-max(start)
        size=np.random.choice(list(range(3,max_size+1)))
        intersecting_options=[]
        for i in range(size):
            g[start[0]+i,start[1]+i]=1 
            if i!=size-1 and i!=0:
                intersecting_options.append((start[0]+i,start[1]+i))
        intersecting=intersecting_options[np.random.choice(np.arange(len(intersecting_options)))]
        g[intersecting[0]-1,intersecting[1]+1]=1
        g[intersecting[0]+1,intersecting[1]-1]=1
        if intersecting[0]-2>=0 and intersecting[1]+2<7 and np.random.choice([0,1]):
            g[intersecting[0]-2,intersecting[1]+2]=1 
            if intersecting[0]-3>=0 and intersecting[1]+3<7 and np.random.choice([0,1]):
                g[intersecting[0]-3,intersecting[1]+3]=1
        if intersecting[1]-2>=0 and intersecting[0]+2<7 and np.random.choice([0,1]):
            g[intersecting[0]+2,intersecting[1]-2]=1 
            if intersecting[1]-3>=0 and intersecting[0]+3<7 and np.random.choice([0,1]):
                g[intersecting[0]+3,intersecting[1]-3]=1 

        return g 

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


gsp_4x4_boards=np.load('data/gsp_4x4_full.npy')
gsp_4x4_probs=np.load('data/gsp_4x4_full_probs.npy')
gsp_4x4_probs=softmax(gsp_4x4_probs)
def gsp_4x4_distribution_sample():
    idx=np.random.choice(np.arange(gsp_4x4_boards.shape[0]),p=gsp_4x4_probs)
    g=gsp_4x4_boards[idx].reshape((4,4)) 
    return g 

def multi_gsp_4x4_sample(n=25):
    idx=np.random.choice(np.arange(gsp_4x4_boards.shape[0]),p=gsp_4x4_probs,size=n,replace=False)
    g=gsp_4x4_boards[idx].reshape((-1,4,4)) 
    return g 

def grammar_4x4_distribution_sample():
    return generate_grid('all',n=4)[0]


def sample_task(name):
    n=7 
    if name=='symmetry':
        gen_function=symmetric_task_distribution_sample
    elif name=='connected':
        gen_function=connected_task_distribution_sample
    elif name=='rectangle':
        gen_function=rectangle_task_distribution_sample
    elif name=='tree':
        gen_function=tree_task_distribution_sample
    elif name=='zigzag':
        gen_function=zigzag_task_distribution_sample 
    elif name=='pyramid':
        gen_function=pyramid_task_distribution_sample
    elif name=='cross':
        gen_function=cross_task_distribution_sample 
    elif name=='copy':
        gen_function=copy_task_distribution_sample
    elif name=='gsp_4x4': 
        gen_function=gsp_4x4_distribution_sample  
        n=4 
    elif name=='all':
        def grammar_gen():
            return generate_grid('all')[0]
        gen_function=grammar_gen 
        
    elif name=='grammar_4x4':
        def grammar_gen():
            return generate_grid('all',n=4)[0]
        gen_function=grammar_gen 
        

    board=gen_function().reshape((n,n))
    while np.sum(board)<3 or np.sum(board)>=40:
        board=gen_function().reshape((n,n))
    choices=np.vstack(np.where(board>0))
    choice=np.random.choice(np.arange(choices.shape[1]))
    start=choices[:,choice] 
    return board,start   