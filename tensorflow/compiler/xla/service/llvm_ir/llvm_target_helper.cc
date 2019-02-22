/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "llvm_target_helper.h"
#include "tensorflow/core/platform/logging.h"


namespace xla {
namespace llvm_ir {

GPUIntrinsics GetIntrinsic(TargetIntrinsicID intrin){

    switch(intrin){
      case kShfl_down_f32:{
         return {llvm::Intrinsic::nvvm_shfl_sync_down_f32, llvm::Intrinsic::not_intrinsic};
       }
      case kShfl_down_i32:{
         return {llvm::Intrinsic::nvvm_shfl_sync_down_i32, llvm::Intrinsic::not_intrinsic};
       }
      case kThread_id_x:{
          GPUIntrinsics rval =  {llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x, llvm::Intrinsic::amdgcn_workitem_id_x};
          return (rval);
       }
      case   kThread_id_y:{
         return {llvm::Intrinsic::nvvm_read_ptx_sreg_tid_y, llvm::Intrinsic::amdgcn_workitem_id_y};
       }
      case  kThread_id_z:{
         return {llvm::Intrinsic::nvvm_read_ptx_sreg_tid_z, llvm::Intrinsic::amdgcn_workitem_id_z};
       }
      case  kBlock_id_x:{
         return {llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x, llvm::Intrinsic::amdgcn_workgroup_id_x};
       }
      case  kBlock_id_y:{
         return {llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_y, llvm::Intrinsic::amdgcn_workgroup_id_y};
       }
      case  kBlock_id_z:{
         return {llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_z, llvm::Intrinsic::amdgcn_workgroup_id_z};
       }
      case  kBarrier_id:{
         return {llvm::Intrinsic::nvvm_barrier0, llvm::Intrinsic::amdgcn_s_barrier};
       }

    }

    return {llvm::Intrinsic::not_intrinsic, llvm::Intrinsic::not_intrinsic};
}

llvm::Intrinsic::ID GetLLVMIntrinsicID(TargetIntrinsicID intrin, 
        llvm::Module* module){
 
  GPUIntrinsics intrin_id = GetIntrinsic(intrin);
  llvm::Triple target_triple = llvm::Triple(module->getTargetTriple());

  llvm::Intrinsic::ID retval = llvm::Intrinsic::not_intrinsic;

  if (target_triple.getArch() == llvm::Triple::nvptx){ 
        retval = intrin_id.nvptx_intrinsic;
  }
  else if  (target_triple.getArch() == llvm::Triple::amdgcn) {
       retval = intrin_id.amdgpu_intrinsic;

  }
  else {

      LOG(FATAL) << "Invalid triple " << target_triple.str();

  }

    return (retval);
}

}  // namespace gpu
}  // namespace xla

