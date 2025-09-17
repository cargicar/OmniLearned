import torch
import torch.nn.functional as F
"""
Implementation of Rectified Flow model. https://arxiv.org/abs/2209.03003
"""
class RectifiedFlow():
  def __init__(self, model=None, num_steps=1000):
    self.model = model
    self.N = num_steps
  
  def get_train_tuple(self, z0=None, z1=None):
    #t = torch.rand((z1.shape[0], 1))
    t = torch.rand(size=(z1.shape[0],)).to(z0.device)
    t0 = t[:, None, None] #TODO double check if this is the right thing to do
    z_t =  t0 * z1 + (1.-t0) * z0
    target = z1 - z0 
        
    return z_t, t, target

  @torch.no_grad()
  def sample_ode(self, z0=None, N=None):
    ### NOTE: Use Euler method to sample from the learned flow
    if N is None:
      N = self.N    
    dt = 1./N
    traj = [] # to store the trajectory
    z = z0.detach().clone()
    batchsize = z.shape[0]
    
    traj.append(z.detach().clone())
    for i in range(N):
      t = torch.ones((batchsize,1)) * i / N
      pred = self.model(z, t)
      z = z.detach().clone() + pred * dt
      
      traj.append(z.detach().clone())

    return traj
 
