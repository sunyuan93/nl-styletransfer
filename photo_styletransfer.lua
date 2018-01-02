require 'torch'
require 'nn'
require 'image'
require 'optim'

require 'loadcaffe'
require 'libcuda_utils'

require 'cutorch'
require 'cunn'

local matio = require 'matio'
local cmd = torch.CmdLine()

-- Basic options
cmd:option('-style_image', 'examples/inputs/seated-nude.jpg', 'Style target image')
cmd:option('-content_image', 'examples/inputs/tubingen.jpg','Content target image')
cmd:option('-style_seg', '', 'Style segmentation')
cmd:option('-style_seg_idxs', '', 'Style seg idxs')
cmd:option('-content_seg', '', 'Content segmentation')
cmd:option('-content_seg_idxs', '', 'Content seg idxs')
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')
-- Optimization options
cmd:option('-num_iterations', 1000)
-- Output options
cmd:option('-print_iter', 1)
cmd:option('-save_iter', 100)
cmd:option('-output_image', 'out.png') 
cmd:option('-index', 1)
cmd:option('-serial', './','serial_example') 

-- Other options
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-backend', 'nn', 'nn|cudnn|clnn')

cmd:option('-mean_layers',  'relu1_1,relu2_1,relu3_1,relu4_1', 'layers for mean')
cmd:option('-mean_weight', 1e6)

cmd:option('-lambda', 1e8) 
cmd:option('-patch', 3)
cmd:option('-eps', 1e-7)
cmd:option('-f_radius', 10)
cmd:option('-f_edge', 0.01)


local function main(params)
  cutorch.setDevice(params.gpu +1)
  cutorch.setHeapTracking(true)
  idx = cutorch.getDevice()
  print('gpu, idx = ', params.gpu, idx)

  --content: pitie transferred input image
  local content_image = image.load(params.content_image, 3)
  local content_image_caffe=preprocess(content_image):float():cuda()
  -- style: target model image
  local style_image = image.load(params.style_image, 3)
  local style_image_caffe = preprocess(style_image):float():cuda()
  
  local mean_layers = params.mean_layers:split(",")
  local c, h, w = content_image:size(1), content_image:size(2), content_image:size(3)
  local _, h2, w2 = style_image:size(1), style_image:size(2), style_image:size(3)
  local index = params.index

  local content_seg = image.load(params.content_seg, 3)
  content_seg = image.scale(content_seg, w, h, 'bilinear')

  local style_seg = image.load(params.style_seg, 3)
  style_seg = image.scale(style_seg, w2, h2, 'bilinear')

  local color_codes = {'black','white','green','red','blue','yellow'}

  local color_content_masks=torch.Tensor(content_seg:size(2),content_seg:size(3)):float():zero()
  local color_style_masks=torch.Tensor(style_seg:size(2),style_seg:size(3)):float():zero()

  for j = 1, #color_codes do
    local content_mask_j = ExtractMask(content_seg, color_codes[j])
    local style_mask_j = ExtractMask(style_seg, color_codes[j])
    color_content_masks:add(content_mask_j:mul(j))
    color_style_masks:add(style_mask_j:mul(j))
  end 

  for i=1,color_content_masks:size(1) do
    for j=1,color_content_masks:size(2) do
      if color_content_masks[i][j]==0 then
        color_content_masks[i][j]=1
      elseif color_content_masks[i][j]>#color_codes then
        color_content_masks[i][j]=#color_codes
      end
    end
  end

  for i=1,color_style_masks:size(1) do
    for j=1,color_style_masks:size(2) do
      if color_style_masks[i][j]==0 then
        color_style_masks[i][j]=1
      elseif color_style_masks[i][j]>#color_codes then
        color_style_masks[i][j]=#color_codes
      end
    end
  end

  local mean_losses={}
  local next_mean_idx=1
  local net = nn.Sequential()
  
  -- load VGG-19 network
  local cnn = loadcaffe.load(params.proto_file, params.model_file, params.backend):float():cuda()
   
  local CSR_fn = 'gen_laplacian/Input_Laplacian_'..tostring(params.patch)..'x'..tostring(params.patch)..'_1e-7_CSR' .. tostring(index) .. '.mat'
  print('loading matting laplacian...', CSR_fn)
  local CSR = matio.load(CSR_fn).CSR:cuda()

  paths.mkdir(tostring(params.serial))
  print('Exp serial:', params.serial)

  for i = 1, #cnn do
    if next_mean_idx <= #mean_layers then
      local layer = cnn:get(i)
      local name = layer.name
      local layer_type = torch.type(layer)
      local is_pooling = (layer_type == 'nn.SpatialMaxPooling' or layer_type == 'cudnn.SpatialMaxPooling')
      net:add(layer)

      if is_pooling then
          color_content_masks = image.scale(color_content_masks, math.ceil(color_content_masks:size(2)/2), math.ceil(color_content_masks:size(1)/2))
          color_style_masks   = image.scale(color_style_masks,   math.ceil(color_style_masks:size(2)/2),   math.ceil(color_style_masks:size(1)/2))
      end 
      
      --color_content_masks=deepcopy(color_content_masks)
      --color_style_masks=deepcopy(color_style_masks)

      if name == mean_layers[next_mean_idx] then
        print("Setting up mean layer  ", i, ":", layer.name)
        local target_features = net:forward(style_image_caffe):clone()
        local mask_target=color_style_masks:clone():cuda()
        local mask_input=color_content_masks:clone():cuda()
        local loss_module = nn.Mean_loss(params.mean_weight, mask_input,mask_target, target_features, color_codes,next_mean_idx):float():cuda()
        net:add(loss_module)
        table.insert(mean_losses,loss_module)
        next_mean_idx = next_mean_idx + 1
      end
    end
  end
  -- We don't need the base CNN anymore, so clean it up to save memory.
  cnn = nil
  for i=1,#net.modules do
    local module = net.modules[i]
    if torch.type(module) == 'nn.SpatialConvolutionMM' then
        -- remove these, not used, but uses gpu memory
        module.gradWeight = nil
        module.gradBias = nil
    end
  end
  collectgarbage()

  local y = net:forward(content_image_caffe)
  local dy = content_image_caffe.new(#y):zero()

  -- Declaring this here lets us access it in maybe_print
  local optim_state = {
      maxIter = params.num_iterations,
      tolX = 0, tolFun = -1,
      verbose=true, 
  }

  local function maybe_print(t, loss)
    local verbose = (params.print_iter > 0 and t % params.print_iter == 0)
    if verbose then
      print(string.format('Iteration %d / %d', t, params.num_iterations))
	  gFile = io.open("loss.txt" ,"a")
	  gFile:write("\r\n") 
      for i, loss_module in ipairs(mean_losses) do
        print(string.format('  Mean %d loss: %f', i, loss_module.loss))
		gFile:write(loss_module.loss)
        gFile:write(" ") 
      end
      print(string.format('  Total loss: %f', loss))  
	  gFile:write(loss) 
	  gFile:close()
    end
  end

  local function maybe_save(t)
    local should_save = params.save_iter > 0 and t % params.save_iter == 0
    should_save = should_save or t == params.num_iterations
    if should_save then
      local disp = deprocess(content_image_caffe:double())
      disp = image.minmax{tensor=disp, min=0, max=1}
      local filename = params.serial .. '/out' .. tostring(index) .. '_t_' .. tostring(t) .. '.png'
      image.save(filename, disp)
    end
  end

  local mean_pixel = torch.CudaTensor({103.939, 116.779, 123.68})
  local meanImage = mean_pixel:view(3, 1, 1):expandAs(content_image_caffe)

  local num_calls = 0

  local content_image_caffe_c=content_image_caffe:clone()
  local function feval(AffineModel) 
    num_calls = num_calls + 1
    
    local output = torch.add(content_image_caffe, meanImage)
    local input  = torch.add(content_image_caffe_c, meanImage)

    net:forward(content_image_caffe)
   
    local gradient_VggNetwork = net:updateGradInput(content_image_caffe, dy)
    
    local gradient_LocalAffine = MattingLaplacian(output, CSR, h, w):mul(params.lambda)
    if num_calls % params.save_iter == 0 then
      local best = SmoothLocalAffine(output, input, params.eps, params.patch, h, w, params.f_radius, params.f_edge)
      fn = params.serial .. '/best' .. tostring(params.index) .. '_t_' .. tostring(num_calls) .. '.png'
      image.save(fn, best)
    end 
    
    local grad = torch.add(gradient_VggNetwork, gradient_LocalAffine)
    --local grad=gradient_VggNetwork
    local loss = 0
    for _, mod in ipairs(mean_losses) do
      loss = loss + mod.loss
    end
    maybe_print(num_calls, loss)
    --maybe_save(num_calls)
    collectgarbage()
    -- optim.lbfgs expects a vector for gradients
    return loss, grad:view(grad:nElement()) 
  end
  
  -- Run optimization.
  local x, losses = optim.lbfgs(feval, content_image_caffe, optim_state)  
end

function MattingLaplacian(output, CSR, h, w)
  local N, c = CSR:size(1), CSR:size(2)
  local CSR_rowIdx = torch.CudaIntTensor(N):copy(torch.round(CSR[{{1,-1},1}]))
  local CSR_colIdx = torch.CudaIntTensor(N):copy(torch.round(CSR[{{1,-1},2}]))
  local CSR_val    = torch.CudaTensor(N):copy(CSR[{{1,-1},3}])

  local output01 = torch.div(output, 256.0)

  local grad = cuda_utils.matting_laplacian(output01, h, w, CSR_rowIdx, CSR_colIdx, CSR_val, N)
  
  grad:div(256.0)
  return grad
end  

function SmoothLocalAffine(output, input, epsilon, patch, h, w, f_r, f_e)
  local output01 = torch.div(output, 256.0)
  local input01 = torch.div(input, 256.0)


  local filter_radius = f_r
  local sigma1, sigma2 = filter_radius / 3, f_e

  local best01= cuda_utils.smooth_local_affine(output01, input01, epsilon, patch, h, w, filter_radius, sigma1, sigma2)
    
  return best01
end 

-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
-- and subtract the mean pixel.
function preprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(256.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img
end

function deepcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[deepcopy(orig_key)] = deepcopy(orig_value)
        end
        setmetatable(copy, deepcopy(getmetatable(orig)))
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

function ExtractMask(seg, color)
  local mask = nil
  if color == 'green' then 
    mask = torch.lt(seg[1], 0.1)
    mask:cmul(torch.gt(seg[2], 1-0.1))
    mask:cmul(torch.lt(seg[3], 0.1))
  elseif color == 'black' then 
    mask = torch.lt(seg[1], 0.1)
    mask:cmul(torch.lt(seg[2], 0.1))
    mask:cmul(torch.lt(seg[3], 0.1))
  elseif color == 'white' then
    mask = torch.gt(seg[1], 1-0.1)
    mask:cmul(torch.gt(seg[2], 1-0.1))
    mask:cmul(torch.gt(seg[3], 1-0.1))
  elseif color == 'red' then 
    mask = torch.gt(seg[1], 1-0.1)
    mask:cmul(torch.lt(seg[2], 0.1))
    mask:cmul(torch.lt(seg[3], 0.1))
  elseif color == 'blue' then
    mask = torch.lt(seg[1], 0.1)
    mask:cmul(torch.lt(seg[2], 0.1))
    mask:cmul(torch.gt(seg[3], 1-0.1))
  elseif color == 'yellow' then
    mask = torch.gt(seg[1], 1-0.1)
    mask:cmul(torch.gt(seg[2], 1-0.1))
    mask:cmul(torch.lt(seg[3], 0.1))
  elseif color == 'grey' then 
    mask = torch.cmul(torch.gt(seg[1], 0.5-0.1), torch.lt(seg[1], 0.5+0.1))
    mask:cmul(torch.cmul(torch.gt(seg[2], 0.5-0.1), torch.lt(seg[2], 0.5+0.1)))
    mask:cmul(torch.cmul(torch.gt(seg[3], 0.5-0.1), torch.lt(seg[3], 0.5+0.1)))
  elseif color == 'lightblue' then
    mask = torch.lt(seg[1], 0.1)
    mask:cmul(torch.gt(seg[2], 1-0.1))
    mask:cmul(torch.gt(seg[3], 1-0.1))
  elseif color == 'purple' then 
    mask = torch.gt(seg[1], 1-0.1)
    mask:cmul(torch.lt(seg[2], 0.1))
    mask:cmul(torch.gt(seg[3], 1-0.1))
  else 
    print('ExtractMask(): color not recognized, color = ', color)
  end 
  return mask:float()
end

-- Undo the above preprocessing.
function deprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img = img + mean_pixel
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):div(256.0)
  return img
end

local Mean_loss, parent = torch.class('nn.Mean_loss', 'nn.Module')

function Mean_loss:__init(strength, mask_input,mask_target, target,color_codes,layer_id)
  parent.__init(self)
  self.mask_input=mask_input
  self.strength = strength
  self.target= target
  self.loss = 0
  self.crit = nn.MSECriterion()
  self.layer_id = layer_id
  self.mask_target = mask_target
  self.color_codes=color_codes
end 

function Mean_loss:updateOutput(input)
  self.output = input
  return self.output
end 

function Mean_loss:updateGradInput(input, gradOutput)
  self.loss = 0
  self.gradInput = gradOutput:clone()
  self.gradInput:zero()
  
  local input_get=input:clone():cuda()
  local target_get=self.target:clone():cuda()

  local num_input=torch.CudaTensor(#self.color_codes):zero()
  local num_target=torch.CudaTensor(#self.color_codes):zero()
  cuda_utils.get_num(self.mask_input,num_input,self.mask_input:size(1),self.mask_input:size(2))
  cuda_utils.get_num(self.mask_target,num_target,self.mask_target:size(1),self.mask_target:size(2))

  local mean_input=torch.CudaTensor(input_get:size(1),#self.color_codes):zero()
  local mean_target=torch.CudaTensor(target_get:size(1),#self.color_codes):zero()
  
  cuda_utils.get_mean(input_get,self.mask_input,mean_input,input_get:size(1),input_get:size(2),input_get:size(3),#self.color_codes)
  cuda_utils.get_mean(target_get,self.mask_target,mean_target,target_get:size(1),target_get:size(2),target_get:size(3),#self.color_codes)
  for i=1,input_get:size(1) do
      mean_input[{i}]:cdiv(num_input)
      mean_target[{i}]:cdiv(num_target)
  end  

  local variance_input=torch.CudaTensor(input_get:size(1),#self.color_codes):zero()
  local variance_target=torch.CudaTensor(target_get:size(1),#self.color_codes):zero()
  
  cuda_utils.get_variance(input_get,self.mask_input,mean_input,variance_input,input_get:size(1),input_get:size(2),input_get:size(3),#self.color_codes)
  cuda_utils.get_variance(target_get,self.mask_target,mean_target,variance_target,target_get:size(1),target_get:size(2),target_get:size(3),#self.color_codes)

  result=torch.CudaTensor(#input_get):zero()
  cuda_utils.transfer(input_get,self.mask_input,mean_input,mean_target,variance_input,variance_target,result,input_get:size(1),input_get:size(2),input_get:size(3),#self.color_codes)
  
  self.loss=self.crit:forward(input,result)
  local dG = self.crit:backward(input,result)
  self.gradInput:add(dG)
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end 

local params = cmd:parse(arg)
main(params)
