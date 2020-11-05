import numpy as np
import torch
import math
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
######################################################

def hidden_layer(in_n,out_n,activation):
    r"""helper function to create a fully-connected neural network layer

    Args:
        in_n (int): number of inputs to the layer; equal to the previous layer's node count
        out_n(int): number of nodes within 
        activation (class): the activation function to be used
        
    Returns:
        neural network layer (class)
    """
    #basic NN layer structure
    return torch.nn.Sequential(
        torch.nn.Linear(in_n, out_n),
        activation)
    
class DR_Network(torch.nn.Module):
    def __init__(self,dim_DR,NN_var):
        """Neural network for a dimension reduction procedure"""
        #NN_var = [dim_in,dim_out,epochs,learning_rate,lambda,penalty,XDR_layer,DRX_layer,DRY_layer]
        ##epochs is ignored

        super(DR_Network,self).__init__()
        self.lr=NN_var[3]
        self.activation = torch.nn.ReLU()

        #reduction bounder + new bounds for BO
        self.activation_DR=torch.nn.Tanh()
        self.DR_range=np.c_[-np.ones(dim_DR),np.ones(dim_DR)]
        
        ###Input (X) to DR###
        self.XDR_size = [NN_var[0], *NN_var[-3]] #dim_in,XDR layers
        self.encoder_DR = torch.nn.Sequential(
            *[hidden_layer(in_n, out_n, self.activation) for in_n, out_n in zip(self.XDR_size, self.XDR_size[1:])],
            torch.nn.Linear(self.XDR_size[-1], dim_DR),
            self.activation_DR
        )

        ###DR to Estimated Output (X_hat)###
        self.DRX_size = [dim_DR,*NN_var[-2]] #dim_DR,DRX layers
        self.decoder_X=torch.nn.Sequential(
            *[hidden_layer(in_n, out_n, self.activation) for in_n, out_n in zip(self.DRX_size, self.DRX_size[1:])],
            torch.nn.Linear(self.DRX_size[-1], NN_var[0])
        )

        ###DR to Estimated Output (Y_hat)###
        self.DRY_size = [dim_DR,*NN_var[-1]] #dim_DR,DRY layers
        self.decoder_Y=torch.nn.Sequential(
            *[hidden_layer(in_n, out_n, self.activation) for in_n, out_n in zip(self.DRY_size, self.DRY_size[1:])],
            torch.nn.Linear(self.DRY_size[-1], NN_var[1])
        )
    
            #define error for the DR goal
        self.error_function = Custom_Loss(NN_var[4],NN_var[5])
        #Adam optimiser
        self.optimiser = torch.optim.Adam(self.parameters(), self.lr)
        self.loss_=[]

    def forward(self, x):
        dr_output=self.encoder_DR(x)
        y_hat=self.decoder_Y(dr_output)
        x_hat=self.decoder_X(dr_output)
        return dr_output,y_hat,x_hat

    def pred_X(self, DR_inputs):
        x_hat=self.decoder_X(DR_inputs)
        return x_hat
 
    def pred_Y(self, DR_inputs):
        y_hat=self.decoder_Y(DR_inputs)
        return y_hat
 
    def reduce_X(self, x):
        dr_output=self.encoder_DR(x)
        return dr_output

class Custom_Loss(torch.nn.Module):

    def __init__(self,lam,P=1):
        super(Custom_Loss,self).__init__()
        self.lam=lam
        self.penalty=P
        self.xhat_history=[]
        self.yhat_history=[]
        self.bound_history=[]

    def forward(self,x_hat,y_hat,train_X,train_Y,xlb,xub,pr=False):
        #use the MSE loss function
        #want the prediction from the original X thorugh NN to be correct
        output1=torch.nn.MSELoss(reduction="sum")(train_X,x_hat)
        #want the prediction from the estimated X thorugh NN to be correct
        output2=torch.nn.MSELoss(reduction="sum")(train_Y,y_hat)*self.lam
        #want to ensure the estimated X is within acceptable bounds of problem
        L1=0
        L2=0
        for d in range(0,x_hat.shape[1]):
            L1+= self.penalty*(torch.clamp(x_hat[:,d]-xlb[d],max=0)**2)
            L2+= self.penalty*(torch.clamp(x_hat[:,d]-xub[d],min=0)**2)
        output3=torch.sum(L1) + torch.sum(L2)
        self.xhat_history.append(output1.item())
        self.yhat_history.append(output2.item())
        self.bound_history.append(output3.item())
        if pr:
            print('x error %s\ny error %s\nboundry error %s'%(output1.item(), output2.item(),output3.item()))
        return output1+output2+output3
    

########################################################################################################################################
########################################################################################################################################
####################################          General Neural Network Architecture          #############################################
########################################################################################################################################
########################################################################################################################################

def conv_layer(in_depth,layer_var,activation):
    r"""helper function to create a convolutional neural network layer

    Args:
        in_depth (int): Number of channels in the input image
        layer_var (list): the necessary variable set to construct the layer:
                [[filter1,kernel_size1,stride1,padding1, pooling kernel_size1,pooling stride1],...,[]]
        activation (class): the activation function to be used

    Returns:
        neural network layer (class)
    """
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_depth,layer_var[0], kernel_size=layer_var[1], stride=layer_var[2], padding=layer_var[3]),
        activation,
        torch.nn.MaxPool2d(kernel_size=layer_var[4],stride=layer_var[5])
        )


def pick_act(act=''):
    r"""helper function to help select a pytorch activation function
    Arg:    
        act (str): 'Tanh' and 'ReLU' at the moment are the only options
    """
    if act=='Tanh':
        return torch.nn.Tanh()
    else:
        return torch.nn.ReLU()


def calc_dim(prev_layer,layer_var):
    r"""
    Args:
        in_n (int): number of inputs to the layer; equal to the previous layer's node count
        out_n(int): number of nodes within 
        activation (class): the activation function to be used
        
    Returns:
        calculated dimensions across the network (array)
    
    Calculations
      Convolutional layer:
        - Input: [D_input,H_input,W_input]
        - output_H = (input_H-k[0]+2P[0]/S[0])+1 
        - output_W = (input_W-k[1]+2P[1]/S[1])+1
        - output_D = filters out (out channel)

      Pooling layer
        - output_H = (input_H-K[0]+2*0)/S[0] + 1
        - output_W = (input_W-K[1]+ 2*0)/S[1] + 1
    """
    D = layer_var[0] # for both
    H = ((prev_layer[1]-layer_var[1][0]+2*layer_var[3][0])/layer_var[2][0])+1  #for conv
    H= ((H-layer_var[4][0])/layer_var[5][0])+1  #for pool
    W = ((prev_layer[2]-layer_var[1][1]+2*layer_var[3][1])/layer_var[2][1])+1 #for conv
    W= ((W-layer_var[4][1])/layer_var[5][1])+1  #for pool

    if all(x%1==0 for x in [D,H,W]):
        return np.array([D,H,W],dtype=int)
    else:
        raise ValueError("Invalid configuration of %s" % [D,H,W])


class Conv_Network(torch.nn.Module):
    def __init__(self,dim_in,dim_out,lr,con_var,fcn_var):
        r"""Convolutional Neural network architecture

        Args:
            dim_in (int): Number of channels in the input image
            dim_out (int): Number of channels produced by the convolution
            lr (float): learning rate for optimization of hyperparameters
            con_var (list of lists): the necessary variable sets to construct the convolutional layers in structure [[layer_1],[layer_2],...]
                Arg for layer_i:
                    filter (int): Number of channels produced by the convolution in the 'depth' direction
                    kernel_size (tuple): Size of the convolving kernel
                    stride (tuple): Stride of the convolution
                    padding (tuple): Padding added to both sides of the input
                    pooling kernel_size (tuple): Size of the pooling kernel. recommended default should be (2,2)
                    pooling stride (tuple): Stride of the pooling kernel. recommended default should be (2,2)

            fcn_var (list): a list containing the number of nodes in each fully-connected layer to construct at the end of the structure

        Returns:
            convolutional neural network class
        """
        
        super(Conv_Network,self).__init__()
        ##if you want to make activation function a selection--> self.activation = pick_act(act)
        self.activation=torch.nn.ReLU()
        self.lr=lr
        self.layer_dim=[dim_in]

        ###Input (X) thru conv layers###
        D_in=[dim_in[0],*[con_var[i][0] for i in range(0,len(con_var))]]
        self.conv = torch.nn.Sequential(
            *[conv_layer(depth, layer, self.activation) for depth, layer in zip(D_in, con_var)]
        )
        for i in range(0,len(con_var)):
            temp=calc_dim(self.layer_dim[-1], con_var[i])
            self.layer_dim=[*self.layer_dim,temp]

        ###Conv Output (C) thru fully conn layer to Estimated Output (Y_hat)###
        self.in_n = [np.prod(self.layer_dim[-1]),*fcn_var] 
        self.fcn=torch.nn.Sequential(
            *[hidden_layer(in_n, out_n, self.activation) for in_n, out_n in zip(self.in_n, self.in_n[1:])])
            
        self.last=torch.nn.Linear(self.in_n[-1], dim_out,bias=True)
    
        #define error
        self.error_function = torch.nn.MSELoss(reduction="sum")
        #define optimiser
        self.optimiser = torch.optim.Adam(self.parameters(), self.lr)
        
    def forward(self, x):
        output=self.conv(x)
        output = output.reshape(output.size(0), -1)
        output=self.fcn(output)
        return self.last(output)
        
    def conv_out(self, x):
        return self.con(x)
        
    def fcn_out(self,x):
        output=self.con(x)
        return self.fcn(output)
          

class FCN_Network(torch.nn.Sequential):

    def __init__(self,dim_in,dim_out,lr,fcn_var):
        r"""Fully-connected or Multiperceptron Neural network architecture

        Args:
            dim_in (int): Number of channels in the input image
            dim_out (int): Number of channels produced by the convolution
            fcn_var (list): a list containing the number of nodes in each fully-connected layer to construct at the end of the structure
            lr (float): the learning rate for the optimization of hyperparameters
        Returns:
            fully-connected neural network class                
        """
        super(FCN_Network, self).__init__()
        ##if you want to make activation function a selection--> self.activation = pick_act(act)
        self.activation = torch.nn.ReLU()
        self.in_n=dim_in
        self.out_n=dim_out
        self.h_n=len(fcn_var)
        self.lr=lr

        ###encode
        self.encoder_n = [dim_in, *fcn_var]
        self.encode = torch.nn.Sequential(*[torch.nn.Sequential(torch.nn.Linear(in_n, out_n),self.activation) for in_n, out_n in zip(self.encoder_n, self.encoder_n[1:])])
        
        ###decode
        self.decode = torch.nn.Linear(self.encoder_n[-1],dim_out, bias=True)

        #define error
        self.error_function = torch.nn.MSELoss(reduction="sum")
        #define optimiser
        self.optimiser = torch.optim.Adam(self.parameters(), self.lr)

    
    def forward(self,x):
        xprj = self.encode(x)  
        y = self.decode(xprj)
        return y

    def encoder(self,x):
        return self.encode(x)
    
    def decoder(self,xprj):
        return self.decode(xprj)


def create_NN(dim_in,dim_out,lr,con_var=[],fcn_var=[]):
    #assumes train x and y are tensors
    if con_var==[]:
        net = FCN_Network(dim_in,dim_out,lr,fcn_var)
    else:
        net = Conv_Network(dim_in,dim_out,lr,con_var,fcn_var)

    return net


def train_NN(train_X,train_Y,epochs,network,pr=False):
    #will use roughly 10% of the training values as a batch for a max of 256; if 10% is less than 2, we don't batch
    # Find optimal model hyperparameters through batching
    if len(train_X)*.1>=2:
        num_batch=min(2**int(math.log(len(train_X)*.1,2)),256)
    else:
        num_batch=len(train_X)
    dataset=Data(torch.as_tensor(train_X),torch.as_tensor(train_Y))
    train_loader = DataLoader(dataset = dataset, batch_size = num_batch, shuffle = True)
    network.train()
    for epoch in range(int(epochs)):
        for batch_idx, (inputs,labels) in enumerate(train_loader):
            network.optimiser.zero_grad()
            #get estimate of y from DR
            y_hat = network(inputs)
            #calc loss
            loss = network.error_function(labels,y_hat)
            loss.backward()
            network.optimiser.step()
            if pr and batch_idx==0 and epoch%100==0: 
                print("loss: %s"%(loss.item()))
    network.eval()



def train_DRNN(train_X,train_Y,epochs,network,x_range, pr=False):
    #will use roughly 10% of the training values as a batch for a max of 256; if 10% is less than 2, we don't batch
    # Find optimal model hyperparameters through batching
    if len(train_X)*.1>=2:
        num_batch=min(2**int(math.log(len(train_X)*.1,2)),256)
    else:
        num_batch=len(train_X)
    dataset=Data(torch.as_tensor(train_X),torch.as_tensor(train_Y))
    train_loader = DataLoader(dataset = dataset, batch_size = num_batch, shuffle = True)
    network.train()
    for epoch in range(int(epochs)):
        for batch_idx, (inputs,labels) in enumerate(train_loader):
            network.optimiser.zero_grad()
            #get estimate of y from DR
            _, y_hat, x_hat = network(inputs)
            loss = network.error_function(
                x_hat, y_hat,
                inputs, labels,
                x_range[:,0],
                x_range[:,1]
                )
            loss.backward()
            network.optimiser.step()
            if pr and batch_idx==0 and epoch%100==0: 
                print("loss: %s"%(loss.item()))
    network.eval()




class Data(Dataset):
    def __init__(self,train_X,train_Y):
        self.x=train_X
        self.y=train_Y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx],self.y[idx]
