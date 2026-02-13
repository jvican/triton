#include <cuda.h>

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

extern "C" {
void *gsanMalloc(ssize_t size, int device, cudaStream_t stream);
void gsanFree(void *ptr, ssize_t size, int device, cudaStream_t stream);
void *gsanGetReservePointer();
size_t gsanGetReserveSize();
}

namespace {

// We use a simple tree of power-of-two sized blocks to manage virtual address
// allocations.
//
// Note that we don't really care about being compact/defragmented in any way,
// since we can reserve millions of times more virtual memory than there is
// physical memory.
// We also are based under the PyTorch CUDACachingAllocator which manages most
// of the hard parts for us and only asks us to allocate large blocks that it
// will divide up as needed.
struct AllocNode {
  CUdeviceptr virtualAddress = 0;
  size_t size = 0;
  CUmemGenericAllocationHandle realHandle = 0;
  CUmemGenericAllocationHandle shadowHandle = 0;
  bool allocated = false;
  AllocNode *parent = nullptr;
  std::unique_ptr<AllocNode> leftChild;
  std::unique_ptr<AllocNode> rightChild;
};

struct AllocatorState {
  CUdeviceptr reserveBaseAddress;
  AllocNode treeRoot;
};

void printCUDAError(CUresult err) {
  const char *errs = "<unknown error>";
  cuGetErrorString(err, &errs);
  fprintf(stderr, "gsan allocator encountered an unexpected error: %s\n", errs);
}

static AllocatorState *alloc = nullptr;
static std::mutex mut;

// Reserve 1 PiB, should be enough for now :)
constexpr size_t kReserveSize = 1ull << 40;
constexpr int kShadowSizeBytes = 8;
constexpr int kShadowMemGranularityBytes = 4;

static_assert((kReserveSize & (kReserveSize - 1)) == 0,
              "kReserveSize must be a power of 2");

size_t cdiv(size_t num, size_t den) { return (num + (den - 1)) / den; }

size_t roundUp(size_t val, size_t alignment) {
  return cdiv(val, alignment) * alignment;
}

CUdeviceptr getShadowAddress(CUdeviceptr virtualAddress) {
  auto shadowBase = alloc->reserveBaseAddress;
  auto realBase = shadowBase + kReserveSize / 2;
  auto byteOffset = virtualAddress - realBase;
  auto wordOffset = byteOffset / kShadowMemGranularityBytes;
  return shadowBase + kShadowSizeBytes * wordOffset;
}

size_t getShadowSize(size_t realMemSize) {
  auto wordSize = cdiv(realMemSize, kShadowMemGranularityBytes);
  return wordSize * kShadowSizeBytes;
}

void splitNode(AllocNode *node) {
  const size_t halfSize = node->size / 2;
  auto left = std::make_unique<AllocNode>();
  auto right = std::make_unique<AllocNode>();

  left->virtualAddress = node->virtualAddress;
  left->size = halfSize;
  left->parent = node;

  right->virtualAddress = node->virtualAddress + halfSize;
  right->size = halfSize;
  right->parent = node;

  node->leftChild = std::move(left);
  node->rightChild = std::move(right);
}

AllocNode *findFreeNode(AllocNode *root, size_t allocSize) {
  std::vector<AllocNode *> stack;
  stack.push_back(root);
  while (!stack.empty()) {
    AllocNode *node = stack.back();
    stack.pop_back();

    if (node->allocated || node->size < allocSize)
      continue;

    if (node->leftChild || node->rightChild) {
      if (node->rightChild)
        stack.push_back(node->rightChild.get());
      if (node->leftChild)
        stack.push_back(node->leftChild.get());
      continue;
    }

    while (node->size > 1 && (node->size / 2) >= allocSize) {
      splitNode(node);
      node = node->leftChild.get();
    }
    return node;
  }
  return nullptr;
}

AllocNode *findNodeByAddress(AllocNode *root, CUdeviceptr address) {
  AllocNode *node = root;
  while (node != nullptr) {
    if (address < node->virtualAddress ||
        address >= node->virtualAddress + node->size)
      return nullptr;

    if (!node->leftChild && !node->rightChild)
      return node;

    if (node->rightChild && address >= node->rightChild->virtualAddress) {
      node = node->rightChild.get();
    } else {
      node = node->leftChild.get();
    }
  }
  return nullptr;
}

bool canCoalesce(const AllocNode *node) {
  if (node == nullptr || node->allocated || !node->leftChild ||
      !node->rightChild)
    return false;

  const auto *left = node->leftChild.get();
  const auto *right = node->rightChild.get();
  const bool leftLeaf = !left->leftChild && !left->rightChild;
  const bool rightLeaf = !right->leftChild && !right->rightChild;
  const bool childrenFree = !left->allocated && !right->allocated;
  return leftLeaf && rightLeaf && childrenFree;
}

void coalesceUp(AllocNode *node) {
  while (node != nullptr && canCoalesce(node)) {
    node->leftChild.reset();
    node->rightChild.reset();
    node = node->parent;
  }
}

int gsanEnsureInit() {
  if (alloc)
    return;

  CUdeviceptr reserveBase;
  CUresult err =
      cuMemAddressReserve(&reserveBase, /*size*/ kReserveSize,
                          /*alignment*/ kReserveSize, /*addr*/ 0, /*flags*/ 0);
  if (err != CUDA_SUCCESS) {
    printCUDAError(err);
    return -1;
  }
  alloc = new AllocatorState();
  alloc->reserveBaseAddress = reserveBase;
  auto *root = &alloc->treeRoot;
  root->virtualAddress = reserveBase + (kReserveSize / 2);
  root->size = kReserveSize / 2;
  return 0;
}

} // namespace

// TODO: Handle streams?
void *gsanMalloc(ssize_t size, int device, CUstream stream) {
  if (size <= 0)
    return nullptr;

  std::lock_guard lg(mut);
  gsanEnsureInit();

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;

  size_t granularity = 0;
  CUresult err = cuMemGetAllocationGranularity(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  if (err != CUDA_SUCCESS) {
    printCUDAError(err);
    return nullptr;
  }
  size_t allocSize = roundUp(static_cast<size_t>(size), granularity);
  AllocNode *node = findFreeNode(&alloc->treeRoot, allocSize);
  if (node == nullptr)
    return nullptr;
  auto shadowSize = getShadowSize(node->size);

  CUmemGenericAllocationHandle realHandle = 0;
  CUmemGenericAllocationHandle shadowHandle = 0;
  bool realMapped = false;
  bool shadowMapped = false;

  CUdeviceptr shadowAddress = getShadowAddress(node->virtualAddress);

  err = cuMemCreate(&realHandle, node->size, &prop, 0);
  if (err != CUDA_SUCCESS)
    goto error;

  err = cuMemCreate(&shadowHandle, getShadowSize(node->size), &prop, 0);
  if (err != CUDA_SUCCESS)
    goto error;

  err = cuMemMap(node->virtualAddress, node->size, /*offset*/ 0, realHandle,
                 /*flags*/ 0);
  if (err != CUDA_SUCCESS)
    goto error;
  realMapped = true;

  err = cuMemMap(shadowAddress, shadowSize, /*offset*/ 0, shadowHandle,
                 /*flags*/ 0);
  if (err != CUDA_SUCCESS)
    goto error;
  shadowMapped = true;

  CUmemAccessDesc accessDesc = {};
  accessDesc.location = prop.location;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  err = cuMemSetAccess(node->virtualAddress, node->size, &accessDesc, 1);
  if (err != CUDA_SUCCESS)
    goto error;

  err = cuMemSetAccess(shadowAddress, shadowSize, &accessDesc, 1);
  if (err != CUDA_SUCCESS)
    goto error;

  // Zero-initialize shadow memory
  err = cuMemsetD8Async(shadowAddress, 0, shadowSize, stream);
  if (err != CUDA_SUCCESS)
    goto error;

  node->realHandle = realHandle;
  node->shadowHandle = shadowHandle;
  node->allocated = true;
  return reinterpret_cast<void *>(node->virtualAddress);

error:
  printCUDAError(err);
  if (shadowMapped)
    cuMemUnmap(shadowAddress, shadowSize);
  if (realMapped)
    cuMemUnmap(node->virtualAddress, node->size);
  if (shadowHandle != 0)
    cuMemRelease(shadowHandle);
  if (realHandle != 0)
    cuMemRelease(realHandle);
  return nullptr;
}

extern "C" void gsanFree(void *ptr, [[maybe_unused]] ssize_t size,
                         [[maybe_unused]] int device,
                         [[maybe_unused]] CUstream stream) {
  if (!ptr)
    return;

  std::lock_guard lg(mut);
  if (alloc == nullptr)
    return;

  auto ptrAddress = reinterpret_cast<CUdeviceptr>(ptr);
  AllocNode *node = findNodeByAddress(&alloc->treeRoot, ptrAddress);
  if (node == nullptr || !node->allocated ||
      node->virtualAddress != ptrAddress) {
    fprintf(stderr, "gsan free called with an invalid pointer\n");
    return;
  }

  const auto shadowAddress = getShadowAddress(node->virtualAddress);
  const auto shadowSize = getShadowSize(node->size);

  CUresult err = cuMemUnmap(node->virtualAddress, node->size);
  if (err != CUDA_SUCCESS)
    printCUDAError(err);

  err = cuMemUnmap(shadowAddress, shadowSize);
  if (err != CUDA_SUCCESS)
    printCUDAError(err);

  err = cuMemRelease(node->realHandle);
  if (err != CUDA_SUCCESS)
    printCUDAError(err);
  node->realHandle = 0;

  err = cuMemRelease(node->shadowHandle);
  if (err != CUDA_SUCCESS)
    printCUDAError(err);
  node->shadowHandle = 0;

  node->allocated = false;
  coalesceUp(node->parent);
}

extern "C" void *gsanGetReservePointer() {
  std::lock_guard lg(mut);
  return alloc->reserveBaseAddress;
}

extern "C" void *gsanGetReserveSize() { return kReserveSize; }
