--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'paths'
require 'optim'
require 'nn'
local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'
local lfs = require "lfs"

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)

-- Create saving directory
lfs.mkdir(opt.savePath)

if opt.testOnly then
   local top1Err, top5Err = trainer:test(0, valLoader)
   print(string.format(' * Results top1: %6.3f  top5: %6.3f', top1Err, top5Err))
   return
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestTop1 = math.huge
local bestTop5 = math.huge
local bestLoss = math.huge
local bestMetric = math.huge
for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   local trainTop1, trainTop5, trainLoss = trainer:train(epoch, trainLoader)

   -- Run model on validation set
   local testTop1, testTop5, testLoss = trainer:test(epoch, valLoader)

   local bestModel = false
   if (opt.bestMetric=='error' and testTop1<bestTop1) or (opt.bestMetric=='loss' and testLoss<bestLoss) then
      bestModel = true
      bestTop1 = testTop1
      bestTop5 = testTop5
      bestLoss = testLoss
      print(string.format(' * Best model (%s):  Loss %1.4f  Top1 %1.4f  Top5 %1.4f ', opt.bestMetric, testLoss, testTop1, testTop5))
   end

   checkpoints.save(epoch, model, trainer.optimState, bestModel, opt)
end

print(string.format(' * Finished:  Loss %6.3f  Top1 %6.3f  Top5 %6.3f ', bestLoss, bestTop1, bestTop5))
