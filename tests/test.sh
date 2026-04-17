timeout 120 python -c "
  import torch, sys                                                                                                                    
  print('1. importing speechbrain...', flush=True)   
  sys.path.insert(0, 'third_party/Mamba-TasNet')                                                                                       
  from speechbrain.lobes.models.dual_path import Dual_Path_Model, Decoder, Encoder                                                     
  print('2. speechbrain OK', flush=True)                                                                                               
                                                                                                                                       
  print('3. patching RMSNorm...', flush=True)        
  import modules.mamba_blocks as _mb                                                                                                   
  if _mb.RMSNorm is None:                                                                                                              
      try:                                
          from mamba_ssm.ops.triton.layer_norm import RMSNorm                                                                          
          _mb.RMSNorm = RMSNorm                                                                                                        
      except ImportError:                                                                                                              
          import torch.nn as nn                                                                                                        
          class _RMSNorm(nn.Module):                                                                                                   
              def __init__(self, hidden_size, eps=1e-6, **kwargs):                                                                     
                  super().__init__()           
                  self.weight = nn.Parameter(torch.ones(hidden_size))                                                                  
                  self.eps = eps                     
              def forward(self, x):                                                                                                    
                  x_f = x.float()                    
                  return (self.weight * x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype)                   
          _mb.RMSNorm = _RMSNorm                                                                                                       
  print('4. RMSNorm:', _mb.RMSNorm, flush=True)                                                                                        
                                                                                                                                       
  from modules.mamba_blocks import MambaBlocksSequential
  print('5. building encoder...', flush=True)                                                                                          
  encoder = Encoder(kernel_size=32, out_channels=256)
  print('6. encoder OK', flush=True)                                                                                                   
                                               
  print('7. building MambaBlocksSequential (intra)...', flush=True)                                                                    
  intra = MambaBlocksSequential(                     
      n_mamba=1, bidirectional=True, d_model=256, d_state=16,                                                                          
      expand=2, d_conv=4, fused_add_norm=False, rms_norm=True,                                                                         
      residual_in_fp32=False, conv_bias=True, bias=False,                                                                              
  )                                                                                                                                    
  print('8. intra OK', flush=True)                                                                                                     
                                                                                                                                       
  print('9. building Dual_Path_Model...', flush=True)
  inter = MambaBlocksSequential(               
      n_mamba=1, bidirectional=True, d_model=256, d_state=16,                                                                          
      expand=2, d_conv=4, fused_add_norm=False, rms_norm=True,
      residual_in_fp32=False, conv_bias=True, bias=False,                                                                              
  )                                                  
  masknet = Dual_Path_Model(                                                                                                           
      num_spks=2, in_channels=256, out_channels=256, 
      num_layers=4, K=400,                                                                                                             
      intra_model=intra, inter_model=inter,    
      norm='ln', linear_layer_after_inter_intra=False,                                                                                 
      skip_around_intra=True,                                                                                                          
  )                                                                                                                                    
  print('10. masknet OK', flush=True)                                                                                                  
                                                                                                                                       
  print('11. building decoder...', flush=True)       
  decoder = Decoder(in_channels=256, out_channels=1, kernel_size=32, stride=16, bias=False)
  print('12. decoder OK', flush=True)                                                                                                  
                                                                                                                                       
  print('13. moving to cuda...', flush=True)                                                                                           
  encoder = encoder.cuda()                                                                                                             
  masknet = masknet.cuda()                           
  decoder = decoder.cuda()                     
  print('14. cuda OK', flush=True)             
                                                                                                                                       
  print('15. forward test...', flush=True)
  x = torch.randn(1, 48000).cuda()                                                                                                     
  mix_w = encoder(x)                                 
  print('16. encoder out:', mix_w.shape, flush=True)                                                                                   
  est_mask = masknet(mix_w)                                                                                                            
  print('17. masknet out: type=', type(est_mask), flush=True)                                                                          
  print('ALL DONE', flush=True)                                                                                                        
  " 2>&1 | tee /hy-tmp/diag.log  