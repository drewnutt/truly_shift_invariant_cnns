import torch
import torch.nn.parallel
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models.circular_pad_layer import circular_pad,circular_pad_3d


class ZeroPad3d(nn.ConstantPad3d):
    def __init__(self, padding):
        super(ZeroPad3d, self).__init__(padding,0)

class ApsPool3d(nn.Module):
    def __init__(self, channels, pad_type='zero', filt_size=3, stride=2, apspool_criterion='l2',
                return_poly_indices=False, N=None,use_first=False):
        super(ApsPool3d, self).__init__()
        
        if stride > 2:
            raise NotImplementedError("original authors only worked out stride==2")
        self.filt_size = filt_size
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))] * 3
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels
        self.N = N
        self.return_poly_indices = return_poly_indices
        
        self.apspool_criterion = apspool_criterion
        self.use_first = use_first
        
        if self.filt_size > 1:
            a = construct_1d_array(self.filt_size)

            filt = torch.Tensor(a[:, None, None]*a[None, :, None]*a[None, None, :])
            filt = filt/torch.sum(filt)
            self.register_buffer('filt', filt[None, None, :, :, :].repeat((self.channels, 1, 1, 1, 1)))

        self.pad = get_pad_layer_3d(pad_type)(self.pad_sizes)
        
        if self.N is not None:
            self.register_buffer('permute_indices',permute_polyphase_3d(self.N+self.N%2).long())
        else:
            self.register_buffer('permute_indices',torch.zeros(48**3).long())
            
            
        
    def forward(self, input_to_pool):
        
        if isinstance(input_to_pool, dict):
            inp, polyphase_indices = input_to_pool['inp'], input_to_pool['polyphase_indices']
    
        else:
#             this is the case when polyphase indices are not pre-defined
            inp = input_to_pool
            polyphase_indices = None

        if self.N is None:
            self.N = inp.shape[2]
            pi = getattr(self,"permute_indices")
            setattr(self,"permute_indices", permute_polyphase_3d(self.N+self.N%2).long().to(pi.device))

        if(self.filt_size == 1):
            return aps_downsample_3d(aps_pad_3d(inp), self.stride, polyphase_indices, return_poly_indices=self.return_poly_indices, permute_indices=self.permute_indices, apspool_criterion=self.apspool_criterion)
        else:
            blurred_inp = F.conv3d(self.pad(inp), self.filt, stride=1, groups=inp.shape[1])
            return aps_downsample_3d(aps_pad_3d(blurred_inp), self.stride, polyphase_indices, return_poly_indices=self.return_poly_indices, permute_indices=self.permute_indices, apspool_criterion=self.apspool_criterion, use_first=self.use_first)

def get_pad_layer_3d(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad3d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad3d
    elif pad_type == 'zero':
        PadLayer = ZeroPad3d
    elif pad_type == 'circular':
        PadLayer = circular_pad_3d
    else:
        raise NotImplementedError('Pad type [%s] not recognized'%pad_type)
    return PadLayer

class ApsPool(nn.Module):
    def __init__(self, channels, pad_type='circular', filt_size=3, stride=2, apspool_criterion = 'l2', 
                return_poly_indices = True, circular_flag = True, N = None):
        super(ApsPool, self).__init__()
        
        self.filt_size = filt_size
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels
        self.N = N
        self.return_poly_indices = return_poly_indices
        
        self.apspool_criterion = apspool_criterion
        
        if self.filt_size>1:
            a = construct_1d_array(self.filt_size)

            filt = torch.Tensor(a[:,None]*a[None,:])
            filt = filt/torch.sum(filt)
            self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)
        
        if self.N is not None:
            self.permute_indices = permute_polyphase(N, stride = 2).cuda()
            
        else:
            self.permute_indices = None
            
            
        
    def forward(self, input_to_pool):
        
        if isinstance(input_to_pool, dict):
            inp, polyphase_indices = input_to_pool['inp'], input_to_pool['polyphase_indices']
    
        else:
#             this is the case when polyphase indices are not pre-defined
            inp = input_to_pool
            polyphase_indices = None

        if(self.filt_size==1):
            return aps_downsample_v2(aps_pad(inp), self.stride, polyphase_indices, return_poly_indices = self.return_poly_indices, permute_indices = self.permute_indices, apspool_criterion = self.apspool_criterion)
            
        else:
            
            blurred_inp = F.conv2d(self.pad(inp), self.filt, stride = 1, groups=inp.shape[1])
            return aps_downsample_v2(aps_pad(blurred_inp), self.stride, polyphase_indices, return_poly_indices = self.return_poly_indices, permute_indices = self.permute_indices, apspool_criterion = self.apspool_criterion)
        

def aps_downsample_3d(x, stride, polyphase_indices = None, return_poly_indices = True, permute_indices = None, apspool_criterion = 'l2',use_first=False):
    
    if stride==1:
        return x
    
    elif stride>2:
        raise Exception('Stride>2 currently not supported in this implementation')
    
    
    else:
        B, C, N, _, _ = x.shape
        N_poly = int(N**3/8) # number of points per polyphase
        Nb2 = int(N/2)

        # permute_indices should never be None
        assert permute_indices is not None
        # if permute_indices is None:
        #     permute_indices = permute_polyphase_3d(N).long().cuda()
        #     self.register_buffer('_permute_indices',permute_polyphase_3d(N+N%2).long())
        #     self.permute_indices = self._permute_indices

        # Flatten the view of each grid
        x = x.view(B, C, -1)

        # Select the points that are in each polyphase (in order, so p0,p1,p2,...)
        # Then order it into polyphase view
        # Then change the order so shape is (batch, polyphases, channels, voxel_info)
        x = torch.index_select(x, dim=2, index=permute_indices).view(B, C, 8, N_poly).permute(0, 2, 1, 3)
        
        if polyphase_indices is None:
            
            polyphase_indices = get_polyphase_indices_3d(x, apspool_criterion,use_first=use_first)
            
        batch_indices = torch.arange(B).to(x.device)
        output = x[batch_indices, polyphase_indices, :, :].view(B, C, Nb2, Nb2, Nb2)
        
        if return_poly_indices:
            return output, polyphase_indices
        else:
            return output

def aps_downsample_v2(x, stride, polyphase_indices = None, return_poly_indices = True, permute_indices = None, apspool_criterion = 'l2'):
    
    if stride==1:
        return x
    
    elif stride>2:
        raise Exception('Stride>2 currently not supported in this implementation')
    
    
    else:
        B, C, N, _ = x.shape
        print( x.shape)
        N_poly = int(N**2/4)
        Nb2 = int(N/2)
        
        if permute_indices is None:
            permute_indices = permute_polyphase(N).long().cuda()
            print(permute_indices)

        x = x.view(B, C, -1)
        x = torch.index_select(x, dim=2, index = permute_indices).view(B, C, 4, N_poly).permute(0, 2, 1, 3)
        
        if polyphase_indices is None:
            
            polyphase_indices = get_polyphase_indices_v2(x, apspool_criterion)
            print(polyphase_indices)
            
        batch_indices = torch.arange(B).cuda()
        output = x[batch_indices, polyphase_indices, :, :].view(B, C, Nb2, Nb2)
        
        if return_poly_indices:
            return output, polyphase_indices
        
        else:
            return output
        
def get_polyphase_indices_v2(x, apspool_criterion):
#     x has the form (B, 4, C, N_poly) where N_poly corresponds to the reduced version of the 2d feature maps

    if apspool_criterion == 'l2':
        norms = torch.norm(x, dim = (2, 3), p = 2)
        polyphase_indices = torch.argmax(norms, dim = 1)
        
    elif apspool_criterion == 'l1':
        norms = torch.norm(x, dim = (2, 3), p = 1)
        polyphase_indices = torch.argmax(norms, dim = 1)
        
    elif apspool_criterion == 'l_infty':
        B = x.shape[0]
        max_vals = torch.max(x.reshape(B, 4, -1).abs(), dim = 2).values
        polyphase_indices = torch.argmax(max_vals, dim = 1)
        
    
    elif apspool_criterion == 'non_abs_max':
        B = x.shape[0]
        max_vals = torch.max(x.reshape(B, 4, -1), dim = 2).values
        polyphase_indices = torch.argmax(max_vals, dim = 1)
        
        
    elif apspool_criterion == 'l2_min':
        norms = torch.norm(x, dim = (2, 3), p = 2)
        polyphase_indices = torch.argmin(norms, dim = 1)
        
    elif apspool_criterion == 'l1_min':
        norms = torch.norm(x, dim = (2, 3), p = 1)
        polyphase_indices = torch.argmin(norms, dim = 1)
        
    else:
        raise Exception('Unknown APS criterion') 

    return polyphase_indices

def get_polyphase_indices_3d(x, apspool_criterion, use_first=False):
#     x has the form (B, [# of polyphases], C, N_poly) where N_poly corresponds to the reduced version of the 2d feature maps

    if type(apspool_criterion) is str:
        apspool_criterion = [apspool_criterion]
    B,polys, _,_ = x.shape
    # Only use the first channel of the image
    if use_first:
        x = x[:,:,0,:].unsqueeze(2)
    norms = torch.zeros((B,polys)).to(x.device)
    if 'l4' in apspool_criterion:
        norms += torch.linalg.vector_norm(x, dim=(2, 3), ord=4)
    if 'l3' in apspool_criterion:
        norms += torch.linalg.vector_norm(x, dim=(2, 3), ord=3)
    if 'l2' in apspool_criterion:
        norms += torch.linalg.vector_norm(x, dim=(2, 3), ord=2)
    if 'l2_mat' in apspool_criterion:
        norms += torch.linalg.norm(x, dim=(2, 3), ord=2)
    if 'l1' in apspool_criterion:
        norms += torch.linalg.vector_norm(x, dim=(2, 3), ord=1)
    if 'l_infty' in apspool_criterion:
        norms += torch.linalg.vector_norm(x, dim=(2,3), ord=float('inf'))
    if 'non_abs_max' in apspool_criterion:
        norms += torch.max(torch.sum(x.reshape(B, polys, -1),dim=2), dim=1).values
    if 'l4_min' in apspool_criterion:
        norms += -torch.linalg.vector_norm(x, dim=(2, 3), ord=4)
    if 'l3_min' in apspool_criterion:
        norms += -torch.linalg.vector_norm(x, dim=(2, 3), ord=3)
    if 'l2_min' in apspool_criterion:
        norms += -torch.linalg.vector_norm(x, dim=(2, 3), ord=2)
    if 'l1_min' in apspool_criterion:
        norms += -torch.linalg.vector_norm(x, dim=(2, 3), ord=1)
    if torch.sum(norms) == 0:
        raise Exception('Unknown APS criterion')

    polyphase_indices = torch.argmax(norms, dim=1)
        
    return polyphase_indices
        
        
def construct_1d_array(filt_size):
    
    if(filt_size==1):
        a = np.array([1.,])
    elif(filt_size==2):
        a = np.array([1., 1.])
    elif(filt_size==3):
        a = np.array([1., 2., 1.])
    elif(filt_size==4):    
        a = np.array([1., 3., 3., 1.])
    elif(filt_size==5):    
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size==6):    
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size==7):    
        a = np.array([1., 6., 15., 20., 15., 6., 1.])
        
    return a
    

def aps_pad(x):
    
    N1, N2 = x.shape[2:4]
    
    if N1%2==0 and N2%2==0:
        return x
    
    if N1%2!=0:
        x = F.pad(x, (0, 0, 0, 1), mode = 'circular')
    
    if N2%2!=0:
        x = F.pad(x, (0, 1, 0, 0), mode = 'circular')
    
    return x
        
def aps_pad_3d(x):
    
    N1, N2, N3 = x.shape[2:5]
    
    if N1%2==0 and N2%2==0 and N3%2==0:
        return x
    
    if N1%2!=0:
        x = F.pad(x, (0, 0, 0, 0, 0, 1), mode = 'circular')
    
    if N2%2!=0:
        x = F.pad(x, (0, 0, 0, 1, 0, 0), mode = 'circular')
    
    if N3%2!=0:
        x = F.pad(x, (0, 1, 0, 0, 0, 0), mode = 'circular')

    return x
    
    
def permute_polyphase(N, stride = 2):
    
    base_even_ind = 2*torch.arange(int(N/2))[None, :]
    base_odd_ind = 1 + 2*torch.arange(int(N/2))[None, :]
    
    even_increment = 2*N*torch.arange(int(N/2))[:,None]
    odd_increment = N + 2*N*torch.arange(int(N/2))[:,None]
    
    p0_indices = (base_even_ind + even_increment).view(-1) # the filter locations if start from 0,0
    p1_indices = (base_even_ind + odd_increment).view(-1) # the filter locations if start from 0,1
    
    p2_indices = (base_odd_ind + even_increment).view(-1) # filter locations if start from 1,0
    p3_indices = (base_odd_ind + odd_increment).view(-1) # filter locations if start from 1,1
    
    permute_indices = torch.cat([p0_indices, p1_indices, p2_indices, p3_indices], dim = 0)
    
    return permute_indices

def permute_polyphase_3d(N):
    # In 3d we have 8 polyphase components
    # shifts for each dimension in {0,1}
    # therefore # polyphase components == # of bit strings of length 3 = 8
    
    base_even_ind = 2*torch.arange(int(N/2))[None, None, :]
    base_odd_ind = 1 + 2*torch.arange(int(N/2))[None, None, :]
    
    even_increment = 2*N*torch.arange(int(N/2))[None, :,None]
    odd_increment = N + 2*N*torch.arange(int(N/2))[None, :,None]
    
    even_square_increment = N ** 2 * 2 *torch.arange(int(N/2))[:, None, None]
    odd_square_increment =  N**2 * (1 + 2 * torch.arange(int(N/2)))[:, None, None]

    p0_indices = (base_even_ind + even_increment + even_square_increment).view(-1)
    p1_indices = (base_even_ind + odd_increment + even_square_increment).view(-1)

    p2_indices = (base_odd_ind + even_increment + even_square_increment).view(-1)
    p3_indices = (base_odd_ind + odd_increment + even_square_increment).view(-1)

    p4_indices = (base_even_ind + even_increment + odd_square_increment).view(-1)
    p5_indices = (base_even_ind + odd_increment + odd_square_increment).view(-1)

    p6_indices = (base_odd_ind + even_increment + odd_square_increment).view(-1)
    p7_indices = (base_odd_ind + odd_increment + odd_square_increment).view(-1)
    
    permute_indices = torch.cat([p0_indices, p1_indices, p2_indices, p3_indices,
                    p4_indices, p5_indices, p6_indices, p7_indices], dim=0)
    
    return permute_indices




def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    elif(pad_type == 'circular'):
        PadLayer = circular_pad
        
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer


def get_pad_layer_1d(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad1d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad1d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad1d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer

# %%
