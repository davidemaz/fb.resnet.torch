--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--

local checkpoint = {}

function checkpoint.latest(opt)
   if opt.resume == 'none' then
      return nil
   end

   local latestPath = paths.concat(opt.resume, 'latest.t7')
   if not paths.filep(latestPath) then
      return nil
   end

   print('=> Loading checkpoint ' .. latestPath)
   local latest = torch.load(latestPath)
   local optimState = torch.load(paths.concat(opt.resume, latest.optimFile))
   return latest, optimState
end

function checkpoint.save(epoch, model, optimState, bestModel, opt)
   -- Don't save the DataParallelTable for easier loading on other machines
   if torch.type(model) == 'nn.DataParallelTable' then
      model = model:get(1)
   end

   local modelFile = opt.savePath .. '/model_' .. epoch .. '.t7'
   local optimFile = opt.savePath .. '/optimState_' .. epoch .. '.t7'
   local modelFile_cur = opt.savePath .. '/model_cur.t7'
   local optimFile_cur = opt.savePath .. '/optimState_cur.t7'
   local latestFile = opt.savePath .. '/latest.t7'
   local bestFile = opt.savePath .. '/model_best.t7'

   if epoch==1 or epoch%opt.saveStep==0 then
      torch.save(modelFile, model)
      torch.save(optimFile, optimState)
   end

   -- Always save current epoch (overwrite)
   torch.save(modelFile_cur, model)
   torch.save(optimFile_cur, optimState)
   torch.save(latestFile, {
      epoch = epoch,
      modelFile = modelFile_cur,
      optimFile = optimFile_cur,
   })

   if bestModel then
      torch.save(bestFile, model)
   end
end

return checkpoint
