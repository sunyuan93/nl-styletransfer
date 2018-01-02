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
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')
-- Optimization options
cmd:option('-num_iterations', 300)
-- Output options
cmd:option('-print_iter', 1)
cmd:option('-save_iter', 50)
cmd:option('-index', 1)
cmd:option('-serial', './','serial_example') 

-- Other options
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-backend', 'nn', 'nn|cudnn|clnn')

cmd:option('-mean_layers',  'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1', 'layers for mean')
cmd:option('-mean_weight', 1e3)

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
  local index = params.index

  local mean_losses={}
  local next_mean_idx=1
  local net = nn.Sequential()
  
  -- load VGG-19 network
  local cnn = loadcaffe.load(params.proto_file, params.model_file, params.backend):float():cuda()

  for i = 1, #cnn do
    if next_mean_idx <= #mean_layers then
      local layer = cnn:get(i)
      local name = layer.name
      net:add(layer)
      if name == mean_layers[next_mean_idx] then
        print("Setting up mean layer  ", i, ":", layer.name)
        local target_features = net:forward(style_image_caffe):clone()
        local loss_module = nn.Mean_loss(params.mean_weight,target_features,next_mean_idx):float():cuda()
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
      for i, loss_module in ipairs(mean_losses) do
        print(string.format('  Mean %d loss: %f', i, loss_module.loss))
      end
      print(string.format('  Total loss: %f', loss))
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

  local num_calls = 0

  local function feval(AffineModel) 
    num_calls = num_calls + 1
    
    net:forward(content_image_caffe)
   
    local gradient_VggNetwork = net:updateGradInput(content_image_caffe, dy)
  
    local grad=gradient_VggNetwork
    local loss = 0
    for _, mod in ipairs(mean_losses) do
      loss = loss + mod.loss
    end
    maybe_print(num_calls, loss)
    maybe_save(num_calls)
    collectgarbage()
    return loss, grad:view(grad:nElement()) 
  end
  
  -- Run optimization.
  local x, losses = optim.lbfgs(feval, content_image_caffe, optim_state)  
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

function Mean_loss:__init(strength,target,layer_id)
  parent.__init(self)
  self.strength = strength
  self.target= target
  self.loss = 0
  self.crit = nn.MSECriterion()
  self.layer_id = layer_id
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
  input_mean=torch.Tensor(input_get:size(1)):zero()
  target_mean=torch.Tensor(target_get:size(1)):zero()

  for i=1,input_get:size(1) do
       input_mean[i]=torch.mean(input_get[i])
       target_mean[i]=torch.mean(target_get[i])
  end
  input_var=torch.Tensor(input_get:size(1)):zero()
  target_var=torch.Tensor(target_get:size(1)):zero()
  for i=1,input_get:size(1) do
        input_var[i]=torch.sum(torch.cmul(input_get[i]-input_mean[i], input_get[i]-input_mean[i]))
        target_var[i]=torch.sum(torch.cmul(target_get[i]-target_mean[i], target_get[i]-target_mean[i]))
  end

  for i=1,input_get:size(1) do
       if input_var[i]>0 then
             input_get[i]:csub(input_mean[i]):mul(target_var[i]/input_var[i]):add(target_mean[i])
       end
  end
  self.loss=self.crit:forward(input,input_get)
  local dG = self.crit:backward(input,input_get)
  self.gradInput:add(dG)
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end 

local params = cmd:parse(arg)
main(params)
