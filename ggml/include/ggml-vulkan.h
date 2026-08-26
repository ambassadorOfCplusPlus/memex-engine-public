#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_VK_NAME "Vulkan"
#define GGML_VK_MAX_DEVICES 16

//GGML_API GGML_CALL void ggml_vk_instance_init(void);

// backend API
GGML_API GGML_CALL ggml_backend_t ggml_backend_vk_init(size_t dev_num);

GGML_API GGML_CALL bool ggml_backend_is_vk(ggml_backend_t backend);
GGML_API GGML_CALL int  ggml_backend_vk_get_device_count(void);
GGML_API GGML_CALL void ggml_backend_vk_get_device_description(int device, char * description, size_t description_size);
GGML_API GGML_CALL void ggml_backend_vk_get_device_memory(int device, size_t * free, size_t * total);

GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num);
// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type(void);

// ---------------------------------------------------------------------------------------
// Opt-in transfer batching.
//
// The backend's set_tensor_async / synchronize interface entries are deliberately left NULL:
// enabling them would hard-abort the fused-MoE expert copies in ggml_backend_sched_copy_inputs,
// which hand set_tensor_async a plain (non-pinned) model-weight pointer, and which are followed
// by a graph_compute with no intervening ggml_backend_synchronize. See the comment on the
// interface table in ggml-vulkan.cpp.
//
// These three give the same fence folding to a caller that explicitly asks for it, and change
// nothing for a caller that does not: they are reached only through this header.
//
// Usage: batch_begin, then N x batch_set_tensor, then batch_end - which submits all recorded
// copies once and waits on one fence. The source pointer MUST be Vulkan-pinned host memory
// (ggml_backend_vk_host_buffer_type), because the shared device staging buffer cannot serve
// several batched copies at once; batch_set_tensor returns false if it is not, having done
// nothing, and the caller should fall back to ggml_backend_tensor_set.
// ---------------------------------------------------------------------------------------
GGML_API GGML_CALL void ggml_backend_vk_batch_begin(ggml_backend_t backend);
GGML_API GGML_CALL bool ggml_backend_vk_batch_set_tensor(ggml_backend_t backend, struct ggml_tensor * tensor,
                                                        const void * data, size_t offset, size_t size);
GGML_API GGML_CALL void ggml_backend_vk_batch_end(ggml_backend_t backend);

// Host pointer to THIS TENSOR'S bytes when the driver placed its buffer in host-visible,
// host-coherent memory (which it does for a small buffer, in the BAR heap), else NULL. Reading a
// few tens of KB straight through the mapping beats ggml_backend_tensor_get's submit+fence,
// which ggml_vk_buffer_read takes on any non-UMA device however small and however host-visible
// the buffer is. Only valid to read once the work producing the tensor has been fenced.
GGML_API GGML_CALL void * ggml_backend_vk_tensor_mapped_ptr(const struct ggml_tensor * tensor);

#ifdef  __cplusplus
}
#endif
