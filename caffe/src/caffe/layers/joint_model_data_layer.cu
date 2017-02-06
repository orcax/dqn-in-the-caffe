#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/net.hpp"

namespace caffe {

template <typename Dtype>
void JointModelDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  this->WaitForInternalThreadToExit();

  // Copy the data
  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
      top[0]->mutable_gpu_data());
  if (this->output_act_) {
    caffe_copy(this->prefetch_act_.count(), this->prefetch_act_.cpu_data(),
               top[act_idx_]->mutable_gpu_data());
  }
  //if (this->output_clip_) {
  //  caffe_copy(this->prefetch_clip_.count(), this->prefetch_clip_.cpu_data(),
  //             top[clip_idx_]->mutable_gpu_data());
  //}
  if (this->output_reward_) {
    caffe_copy(this->prefetch_reward_.count(), this->prefetch_reward_.cpu_data(),
               top[reward_idx_]->mutable_gpu_data());
  }
  if (this->output_value_) {
    caffe_copy(this->prefetch_value_.count(), this->prefetch_value_.cpu_data(),
               top[value_idx_]->mutable_gpu_data());
  }
  //if (this->output_life_) {
  //  caffe_copy(this->prefetch_life_.count(), this->prefetch_life_.cpu_data(),
  //             top[life_idx_]->mutable_gpu_data());
  //}

  // Start a new prefetch thread
  this->StartInternalThread();
}

INSTANTIATE_LAYER_GPU_FORWARD(JointModelDataLayer);

}  // namespace caffe
