import numpy as np


class RNN:
    #hidden size: size of a single hidden layer
    def __init__(self,n_x,n_h,seq_len,eta):

        self.n_x=n_x
        self.n_h=n_h
        self.seq_len=seq_len
        self.eta=eta
        self.b=np.zeros(shape=(n_h,1))
        self.c=np.zeros(shape=(n_x,1))
        self.U=np.random.uniform(low=0,high=1,size=(n_h,n_x))
        self.V=np.random.uniform(low=0,high=1,size=(n_x,n_h)) #input length= output length
        self.W=np.random.uniform(low=0,high=1,size=(n_h,n_h))

        self.mU = np.zeros_like(self.U)
        self.mW = np.zeros_like(self.W)
        self.mV = np.zeros_like(self.V)
        self.mb = np.zeros_like(self.b)
        self.mc = np.zeros_like(self.c)
    

    def forward(self,inputs,targets,s_prev):
        #s_prev - initial hidden state
        x,s,o,y_hat={},{},{},{}
        s[-1]=np.copy(s_prev)
        loss=0
        
        for t in range(len(inputs)):

            x[t]=np.zeros((self.n_x,1))
            x[t][inputs[t]]=1
            s[t]=np.tanh(self.W@s[t-1] + self.U@x[t]+self.b)
            o[t]=self.V@s[t] + self.c
            y_hat[t]=np.exp(o[t])/np.sum(np.exp(o[t]))

            loss=loss- np.log(y_hat[t][targets[t],0])

        return loss,x,s,y_hat

    def backprop(self,x,s,y_hat,targets):
        dU,dV,dW=np.zeros_like(self.U), np.zeros_like(self.V), np.zeros_like(self.W)
        db, dc = np.zeros_like(self.b), np.zeros_like(self.c)
        ds_next=np.zeros_like(s[0])
        for t in reversed(range(self.seq_len)):
            do=np.copy(y_hat[t])
            do[targets[t]]=do[targets[t]]-1 #wrt output units

            dV+= do @ s[t].T #wrt parameter V
            dc+=do

            ds=self.V.T@do + ds_next
            ds_raw= (1-s[t]*s[t])*ds  # 1-tanh^2 * ds
            db=db + ds_raw

            dU+=ds_raw @ x[t].T
            dW+=ds_raw @ s[t-1].T
            ds_next=self.W.T @ ds_raw
        
        for dpara in [dU,dW,dV,db,dc]:
            np.clip(dpara, -5, 5, out = dpara)
        
        return dU,dW,dV,db,dc

    def update_params(self, dU, dW, dV, db, dc):
        for para, dpara, mem in zip(['U', 'W', 'V', 'b', 'c'], 
                                    [dU, dW, dV, db, dc], 
                                    ['mU', 'mW', 'mV', 'mb', 'mc']):
            
            setattr(self, mem, getattr(self, mem) + dpara * dpara)
            setattr(self, para, getattr(self, para) - self.eta * dpara/np.sqrt(getattr(self, mem) + 1e-8))


    def train(self,inputs,char_to_int,int_to_char,max_iter=1e4):
        iter=0
        pos=0
        loss_list=[]
        loss_list.append(- np.log(1 / self.n_x) * self.seq_len)      

        while iter <=max_iter:
            if iter%5==0:
                print(iter)
            # rest rnn after an epoch

            if pos + self.seq_len + 1 >=len(inputs) or iter==0:
                s_prev=np.zeros((self.n_h,1))
                pos=0
            
            #chars to int
            input_batch=[char_to_int[ch] for ch in inputs[pos:pos+self.seq_len]] #list of integers
            target_batch=[char_to_int[ch] for ch in inputs[pos+1:pos+self.seq_len+1]]
            pos=pos+1

            #forward pass

            loss,x,s,y_hat=self.forward(inputs=input_batch,targets=target_batch,s_prev=s_prev)
            loss_list.append(loss_list[-1] * 0.999 + loss * 0.001)

            ##backprop
            dU,dW,dV,db,dc=self.backprop(x,s,y_hat,target_batch)

            self.update_params(dU, dW, dV, db, dc)
            s_prev=s[self.seq_len-1]
            iter = iter + 1
                
        sample_ix = self.make_sample(s_prev, target_batch[-1], 200)
        sample_char = ''.join(int_to_char[ix] for ix in sample_ix)
        
        return loss_list, sample_char





  

        
    def make_sample(self, hprev, seed_ix, n):
        """
        sample a length n sequence from the model
        """
        x = np.zeros((self.n_x, 1))
        x[seed_ix] = 1
        ixes = []
        h = np.copy(hprev)
        
        for t in range(n):
            h = np.tanh(self.U @ x + self.W @ h + self.b)
            y = self.V @ h + self.c
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(self.n_x), p = p.ravel())
            x = np.zeros((self.n_x, 1))
            x[ix] = 1
            ixes.append(ix)
        return ixes  





with open("input.txt","r") as f:
    words=f.read()

chars=list(set(words))

char_to_int={char:i for i,char in enumerate(chars)}
int_to_char={i:char for i,char in enumerate(chars)}

word_size=len(words)
vocab_size=len(chars)
rnn=RNN(n_x=word_size,n_h=100,seq_len=25,eta=1e-1) #seq_length -> processes 25 characters at a time

loss_list,sample_char=rnn.train(words,char_to_int,int_to_char,max_iter=50000)