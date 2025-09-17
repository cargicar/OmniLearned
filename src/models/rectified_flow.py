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
    t = torch.rand((z1.shape[0], 1))
    z_t =  t * z1 + (1.-t) * z0
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
  

# Initialize random noise tensor
x = torch.randn(64, 1000, 4).to(device)

# Set the number of steps for the solver
num_steps = 100  # A lower number of steps is a key advantage of RF
# Discretize the time from 0 to 1
dt = 1.0 / num_steps
times = torch.arange(0, 1, dt).to(device)


with torch.no_grad():
    for t_step in times:
        t = t_step.repeat(x.shape[0])[:, None] # Adjust shape for model input
        
        # Predict the velocity field with your model
        # The model's body takes the noisy data and conditions
        z_body = model.body(x, cond, pid, add_info, t)
        # The generator predicts the velocity
        v = model.generator(z_body, y, gap, energy)
        
        # Euler integration step
        x = x + v * dt

# The final result after the loop is your generated point cloud
generated_point_cloud = x  
def train_rectified_flow(rectified_flow, optimizer, pairs, batchsize, inner_iters):
  loss_curve = []
  for i in range(inner_iters+1):
    optimizer.zero_grad()
    indices = torch.randperm(len(pairs))[:batchsize]
    batch = pairs[indices]
    z0 = batch[:, 0].detach().clone()
    z1 = batch[:, 1].detach().clone()
    z_t, t, target = rectified_flow.get_train_tuple(z0=z0, z1=z1)

    pred = rectified_flow.model(z_t, t)
    loss = (target - pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1)
    loss = loss.mean()
    loss.backward()
    
    optimizer.step()
    loss_curve.append(np.log(loss.item())) ## to store the loss curve

  return rectified_flow, loss_curve

