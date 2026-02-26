/*
 * d3d12_compute.c — Native D3D12 compute backend for WSL2
 *
 * Bypasses Vulkan/Dozen translation layer by calling libd3d12.so directly.
 * Compiled with gcc using directx-headers-dev (WSL2 stubs).
 *
 * Build: bash build_d3d12.sh
 */

#define COBJMACROS
#define INITGUID
#define CINTERFACE

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <dlfcn.h>
#include <pthread.h>
#include <time.h>

/* WSL2 stubs — must include unknwnbase.h before d3d12.h to get IUnknown */
#include <wsl/stubs/unknwnbase.h>

/* DirectX headers */
#include <directx/d3d12.h>
#include <directx/d3dcommon.h>

/* ============================================================================
 * DXCore — manual C vtable definitions (headers are C++ only)
 * ============================================================================ */

/* GUIDs for DXCore */
static const GUID DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS_ = {
    0x0c9ece4d, 0x2f6e, 0x4f01,
    {0x8c, 0x96, 0xe8, 0x9e, 0x33, 0x1b, 0x47, 0xb1}
};

static const GUID IID_IDXCoreAdapterFactory_ = {
    0x78ee5945, 0xc36e, 0x4b13,
    {0xa6, 0x69, 0x00, 0x5d, 0xd1, 0x1c, 0x0f, 0x06}
};

static const GUID IID_IDXCoreAdapterList_ = {
    0x526c7776, 0x40e9, 0x459b,
    {0xb7, 0x11, 0xf3, 0x2a, 0xd7, 0x6d, 0xfc, 0x28}
};

static const GUID IID_IDXCoreAdapter_ = {
    0xf0db4c7f, 0xfe5a, 0x42a2,
    {0xbd, 0x62, 0xf2, 0xa6, 0xcf, 0x6f, 0xc8, 0x3e}
};

/* DXCore property enum values (from dxcore_interface.h) */
#define DXCORE_ADAPTER_PROPERTY_InstanceLuid         0
#define DXCORE_ADAPTER_PROPERTY_DriverDescription    2
#define DXCORE_ADAPTER_PROPERTY_IsHardware          11

/* DXCore vtable structures — minimal for adapter enumeration */
typedef struct IDXCoreAdapter IDXCoreAdapter;
typedef struct IDXCoreAdapterList IDXCoreAdapterList;
typedef struct IDXCoreAdapterFactory IDXCoreAdapterFactory;

typedef struct IDXCoreAdapterVtbl {
    /* IUnknown */
    HRESULT (*QueryInterface)(IDXCoreAdapter *This, REFIID riid, void **ppv);
    ULONG (*AddRef)(IDXCoreAdapter *This);
    ULONG (*Release)(IDXCoreAdapter *This);
    /* IDXCoreAdapter */
    HRESULT (*IsValid)(IDXCoreAdapter *This);
    HRESULT (*IsAttributeSupported)(IDXCoreAdapter *This, REFIID attr);
    HRESULT (*IsPropertySupported)(IDXCoreAdapter *This, uint32_t property);
    HRESULT (*GetProperty)(IDXCoreAdapter *This, uint32_t property, size_t bufSize, void *buf);
    HRESULT (*GetPropertySize)(IDXCoreAdapter *This, uint32_t property, size_t *size);
    HRESULT (*IsQueryStateSupported)(IDXCoreAdapter *This, uint32_t state);
    HRESULT (*QueryState)(IDXCoreAdapter *This, uint32_t state, size_t inSize, const void *in, size_t outSize, void *out);
    HRESULT (*IsSetStateSupported)(IDXCoreAdapter *This, uint32_t state);
    HRESULT (*SetState)(IDXCoreAdapter *This, uint32_t state, size_t inSize, const void *in, size_t outSize, void *out);
    HRESULT (*GetFactory)(IDXCoreAdapter *This, REFIID riid, void **ppv);
} IDXCoreAdapterVtbl;

struct IDXCoreAdapter { const IDXCoreAdapterVtbl *lpVtbl; };

typedef struct IDXCoreAdapterListVtbl {
    /* IUnknown */
    HRESULT (*QueryInterface)(IDXCoreAdapterList *This, REFIID riid, void **ppv);
    ULONG (*AddRef)(IDXCoreAdapterList *This);
    ULONG (*Release)(IDXCoreAdapterList *This);
    /* IDXCoreAdapterList */
    HRESULT (*GetAdapter)(IDXCoreAdapterList *This, uint32_t index, REFIID riid, void **ppv);
    uint32_t (*GetAdapterCount)(IDXCoreAdapterList *This);
    HRESULT (*IsStale)(IDXCoreAdapterList *This);
    HRESULT (*GetFactory)(IDXCoreAdapterList *This, REFIID riid, void **ppv);
    HRESULT (*Sort)(IDXCoreAdapterList *This, uint32_t numPrefs, const uint32_t *prefs);
    HRESULT (*IsAdapterPreferenceSupported)(IDXCoreAdapterList *This, uint32_t pref);
} IDXCoreAdapterListVtbl;

struct IDXCoreAdapterList { const IDXCoreAdapterListVtbl *lpVtbl; };

typedef struct IDXCoreAdapterFactoryVtbl {
    /* IUnknown */
    HRESULT (*QueryInterface)(IDXCoreAdapterFactory *This, REFIID riid, void **ppv);
    ULONG (*AddRef)(IDXCoreAdapterFactory *This);
    ULONG (*Release)(IDXCoreAdapterFactory *This);
    /* IDXCoreAdapterFactory */
    HRESULT (*CreateAdapterList)(IDXCoreAdapterFactory *This,
        uint32_t numAttrs, const GUID *attrs, REFIID riid, void **ppv);
    HRESULT (*GetAdapterByLuid)(IDXCoreAdapterFactory *This,
        const LUID *luid, REFIID riid, void **ppv);
    HRESULT (*IsNotificationTypeSupported)(IDXCoreAdapterFactory *This, uint32_t type);
    HRESULT (*RegisterEventNotification)(IDXCoreAdapterFactory *This,
        void *obj, uint32_t type, void *cb, void *ctx, uint32_t *cookie);
    HRESULT (*UnregisterEventNotification)(IDXCoreAdapterFactory *This, uint32_t cookie);
} IDXCoreAdapterFactoryVtbl;

struct IDXCoreAdapterFactory { const IDXCoreAdapterFactoryVtbl *lpVtbl; };

/* ============================================================================
 * SPIR-V → DXIL compilation types (from spirv_to_dxil.h, inlined to avoid
 * pulling in Mesa internal headers)
 * ============================================================================ */

typedef enum {
    DXIL_SPIRV_SHADER_COMPUTE_ = 5,
} dxil_spirv_shader_stage_;

typedef enum {
    NO_DXIL_VALIDATION_ = 0,
    DXIL_VALIDATOR_1_4_ = 0x10004,
} dxil_validator_version_;

typedef enum {
    SHADER_MODEL_6_0_ = 0x60000,
    SHADER_MODEL_6_2_ = 0x60002,
} dxil_shader_model_;

typedef enum {
    DXIL_SPIRV_YZ_FLIP_NONE_ = 0,
} dxil_spirv_yz_flip_mode_;

typedef enum {
    DXIL_SPIRV_SYSVAL_TYPE_ZERO_ = 0,
} dxil_spirv_sysval_type_;

struct spirv_to_dxil_metadata {
    bool requires_runtime_data;
    bool needs_draw_sysvals;
};

struct spirv_to_dxil_object {
    struct spirv_to_dxil_metadata metadata;
    struct {
        void *buffer;
        size_t size;
    } binary;
};

struct spirv_to_dxil_debug_options {
    bool dump_nir;
};

struct spirv_to_dxil_runtime_conf {
    struct { uint32_t register_space; uint32_t base_shader_register; } runtime_data_cbv;
    struct { uint32_t register_space; uint32_t base_shader_register; } push_constant_cbv;
    uint32_t first_vertex_and_base_instance_mode;
    uint32_t workgroup_id_mode;
    struct { uint32_t mode; uint16_t y_mask; uint16_t z_mask; } yz_flip;
    bool declared_read_only_images_as_srvs;
    bool inferred_read_only_images_as_srvs;
    bool force_sample_rate_shading;
    bool lower_view_index;
    bool lower_view_index_to_rt_layer;
    uint32_t shader_model_max;
};

struct spirv_to_dxil_logger {
    void *priv;
    void (*log)(void *priv, const char *msg);
};

typedef bool (*PFN_spirv_to_dxil)(
    const uint32_t *words, size_t word_count,
    void *specializations, unsigned int num_specializations,
    uint32_t stage, const char *entry_point_name,
    uint32_t validator_version_max,
    const struct spirv_to_dxil_debug_options *debug_options,
    const struct spirv_to_dxil_runtime_conf *conf,
    const struct spirv_to_dxil_logger *logger,
    struct spirv_to_dxil_object *out_dxil);

typedef void (*PFN_spirv_to_dxil_free)(struct spirv_to_dxil_object *dxil);

/* ============================================================================
 * ID3D12Device2 — Pipeline State Stream API (used by Dozen driver)
 * Uses types from d3d12.h: ID3D12Device2, D3D12_PIPELINE_STATE_STREAM_DESC,
 * D3D12_PIPELINE_STATE_SUBOBJECT_TYPE enum, IID_ID3D12Device2
 * ============================================================================ */

/* Pipeline state stream subobject structs for compute.
 * Each subobject is: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE (uint32) + padding + payload.
 * Aligned to pointer size (8 bytes on 64-bit). */
typedef struct {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type;  /* ROOT_SIGNATURE = 0 */
    uint32_t _pad;
    ID3D12RootSignature *root_sig;
} d3d12_pss_root_sig;

typedef struct {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type;  /* CS = 6 */
    uint32_t _pad;
    D3D12_SHADER_BYTECODE cs;
} d3d12_pss_cs;

typedef struct {
    d3d12_pss_root_sig root_sig;
    d3d12_pss_cs cs;
} d3d12_compute_stream;

/* ============================================================================
 * Function pointers (loaded via dlopen)
 * ============================================================================ */

typedef HRESULT (*PFN_D3D12CreateDevice)(IUnknown *pAdapter, uint32_t MinFeatureLevel,
    REFIID riid, void **ppDevice);
typedef HRESULT (*PFN_D3D12SerializeRootSignature)(
    const D3D12_ROOT_SIGNATURE_DESC *pRootSignature,
    D3D_ROOT_SIGNATURE_VERSION Version, ID3D10Blob **ppBlob, ID3D10Blob **ppErrorBlob);
typedef HRESULT (*PFN_DXCoreCreateAdapterFactory)(REFIID riid, void **ppv);

/* ============================================================================
 * Global state
 * ============================================================================ */

#define MAX_HANDLES 4096

typedef struct {
    enum { HANDLE_NONE = 0, HANDLE_BUFFER, HANDLE_PIPELINE } type;
    union {
        ID3D12Resource *resource;
        struct {
            ID3D12PipelineState *pso;
            ID3D12RootSignature *root_sig;
            uint32_t num_srvs;
            uint32_t num_uavs;
            uint32_t num_cbvs;
        } pipeline;
    };
} HandleEntry;

static struct {
    bool initialized;

    /* Libraries */
    void *lib_d3d12;
    void *lib_dxcore;
    void *lib_spirv2dxil;

    /* Function pointers */
    PFN_D3D12CreateDevice pfn_D3D12CreateDevice;
    PFN_D3D12SerializeRootSignature pfn_D3D12SerializeRootSignature;
    PFN_DXCoreCreateAdapterFactory pfn_DXCoreCreateAdapterFactory;
    PFN_spirv_to_dxil pfn_spirv_to_dxil;
    PFN_spirv_to_dxil_free pfn_spirv_to_dxil_free;

    /* D3D12 objects */
    ID3D12Device *device;
    ID3D12CommandQueue *queue;
    ID3D12CommandAllocator *allocator;
    ID3D12GraphicsCommandList *cmdlist;
    ID3D12Fence *fence;
    uint64_t fence_value;
    HANDLE fence_event;

    /* Descriptor heap for compute shader UAV/CBV bindings */
    ID3D12DescriptorHeap *desc_heap;
    uint32_t desc_size;       /* increment size for CBV_SRV_UAV */
    uint32_t desc_heap_offset; /* current allocation offset in heap */
#define DESC_HEAP_SIZE 65536

    /* Handle table */
    HandleEntry handles[MAX_HANDLES];
    uint32_t next_handle;

    /* Adapter info */
    char adapter_name[256];
} g;

/* ============================================================================
 * Error reporting
 * ============================================================================ */

static char g_last_error[512] = {0};

static void set_error(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_last_error, sizeof(g_last_error), fmt, args);
    va_end(args);
    fprintf(stderr, "[d3d12c] %s\n", g_last_error);
}

/* ============================================================================
 * Handle management
 * ============================================================================ */

static uint64_t alloc_handle(void) {
    for (uint32_t i = g.next_handle; i < MAX_HANDLES; i++) {
        if (g.handles[i].type == HANDLE_NONE) {
            g.next_handle = i + 1;
            return (uint64_t)(i + 1); /* 1-based so 0 = invalid */
        }
    }
    /* Wrap around */
    for (uint32_t i = 1; i < g.next_handle && i < MAX_HANDLES; i++) {
        if (g.handles[i].type == HANDLE_NONE) {
            g.next_handle = i + 1;
            return (uint64_t)(i + 1);
        }
    }
    return 0;
}

static HandleEntry* get_handle(uint64_t h) {
    if (h == 0 || h > MAX_HANDLES) return NULL;
    HandleEntry *e = &g.handles[h - 1];
    if (e->type == HANDLE_NONE) return NULL;
    return e;
}

/* ============================================================================
 * Fence helper: Wait for GPU to finish
 * ============================================================================ */

/* WSL2 fence events: We use a simple polling approach since Windows HANDLE
 * events don't work the same in WSL2. The fence value comparison is sufficient. */

static HRESULT wait_for_gpu(void) {
    g.fence_value++;
    HRESULT hr = ID3D12CommandQueue_Signal(g.queue, g.fence, g.fence_value);
    if (FAILED(hr)) return hr;

    uint64_t completed = ID3D12Fence_GetCompletedValue(g.fence);
    if (completed < g.fence_value) {
        /* Set event and wait */
        hr = ID3D12Fence_SetEventOnCompletion(g.fence, g.fence_value, g.fence_event);
        if (FAILED(hr)) return hr;

        /* Spin-wait on fence value (WSL2 compatible) */
        while (ID3D12Fence_GetCompletedValue(g.fence) < g.fence_value) {
            /* Yield to avoid burning CPU */
            struct timespec ts = {0, 100000}; /* 100us */
            nanosleep(&ts, NULL);
        }
    }
    return S_OK;
}

/* ============================================================================
 * Public API: Initialization
 * ============================================================================ */

int d3d12c_init(void) {
    if (g.initialized) return 0;
    memset(&g, 0, sizeof(g));

    /* Load libraries */
    g.lib_d3d12 = dlopen("libd3d12.so", RTLD_NOW);
    if (!g.lib_d3d12) {
        set_error("Failed to load libd3d12.so: %s", dlerror());
        return -1;
    }

    g.lib_dxcore = dlopen("libdxcore.so", RTLD_NOW);
    if (!g.lib_dxcore) {
        set_error("Failed to load libdxcore.so: %s", dlerror());
        return -1;
    }

    /* Try multiple paths for spirv_to_dxil (optional — only needed for shader compilation) */
    g.lib_spirv2dxil = dlopen("libspirv_to_dxil.so", RTLD_NOW);
    if (!g.lib_spirv2dxil)
        g.lib_spirv2dxil = dlopen("/home/mukshud/mesa-dozen-install/lib/aarch64-linux-gnu/libspirv_to_dxil.so", RTLD_NOW);
    if (!g.lib_spirv2dxil) {
        fprintf(stderr, "[d3d12c] Warning: libspirv_to_dxil.so not found, SPIR-V compilation disabled\n");
    }

    /* Resolve D3D12/DXCore function pointers */
    g.pfn_D3D12CreateDevice = (PFN_D3D12CreateDevice)dlsym(g.lib_d3d12, "D3D12CreateDevice");
    g.pfn_D3D12SerializeRootSignature = (PFN_D3D12SerializeRootSignature)dlsym(g.lib_d3d12, "D3D12SerializeRootSignature");
    g.pfn_DXCoreCreateAdapterFactory = (PFN_DXCoreCreateAdapterFactory)dlsym(g.lib_dxcore, "DXCoreCreateAdapterFactory");

    if (!g.pfn_D3D12CreateDevice || !g.pfn_D3D12SerializeRootSignature ||
        !g.pfn_DXCoreCreateAdapterFactory) {
        set_error("Failed to resolve D3D12/DXCore functions");
        return -1;
    }

    /* Enable experimental shader models — required on WSL2 to accept Mesa's
     * unsigned DXIL bytecode. Without this, CreateComputePipelineState returns
     * E_INVALIDARG (0x80070057) for all shaders compiled by spirv_to_dxil.
     * Must be called BEFORE D3D12CreateDevice. */
    {
        typedef HRESULT (*PFN_D3D12EnableExperimentalFeatures)(
            UINT, const GUID*, void*, UINT*);
        PFN_D3D12EnableExperimentalFeatures pfn_enable =
            (PFN_D3D12EnableExperimentalFeatures)dlsym(g.lib_d3d12,
                "D3D12EnableExperimentalFeatures");
        if (pfn_enable) {
            HRESULT ehr = pfn_enable(1, &D3D12ExperimentalShaderModels, NULL, NULL);
            if (SUCCEEDED(ehr)) {
                fprintf(stderr, "[d3d12c] Experimental shader models enabled\n");
            } else {
                fprintf(stderr, "[d3d12c] Warning: D3D12EnableExperimentalFeatures failed: 0x%08x\n",
                    (unsigned)ehr);
            }
        }
    }

    /* Resolve spirv_to_dxil (optional) */
    if (g.lib_spirv2dxil) {
        g.pfn_spirv_to_dxil = (PFN_spirv_to_dxil)dlsym(g.lib_spirv2dxil, "spirv_to_dxil");
        g.pfn_spirv_to_dxil_free = (PFN_spirv_to_dxil_free)dlsym(g.lib_spirv2dxil, "spirv_to_dxil_free");
        if (!g.pfn_spirv_to_dxil || !g.pfn_spirv_to_dxil_free) {
            fprintf(stderr, "[d3d12c] Warning: spirv_to_dxil symbols not found\n");
            g.pfn_spirv_to_dxil = NULL;
            g.pfn_spirv_to_dxil_free = NULL;
        }
    }

    /* Enumerate adapters via DXCore */
    IDXCoreAdapterFactory *factory = NULL;
    HRESULT hr = g.pfn_DXCoreCreateAdapterFactory(&IID_IDXCoreAdapterFactory_, (void**)&factory);
    if (FAILED(hr)) {
        set_error("DXCoreCreateAdapterFactory failed: 0x%08x", (unsigned)hr);
        return -1;
    }

    IDXCoreAdapterList *list = NULL;
    hr = factory->lpVtbl->CreateAdapterList(factory, 1,
        &DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS_, &IID_IDXCoreAdapterList_, (void**)&list);
    if (FAILED(hr)) {
        factory->lpVtbl->Release(factory);
        set_error("CreateAdapterList failed: 0x%08x", (unsigned)hr);
        return -1;
    }

    uint32_t adapter_count = list->lpVtbl->GetAdapterCount(list);
    if (adapter_count == 0) {
        list->lpVtbl->Release(list);
        factory->lpVtbl->Release(factory);
        set_error("No D3D12 adapters found");
        return -1;
    }

    /* Pick first hardware adapter */
    IDXCoreAdapter *adapter = NULL;
    IUnknown *adapter_unknown = NULL;
    for (uint32_t i = 0; i < adapter_count; i++) {
        IDXCoreAdapter *a = NULL;
        hr = list->lpVtbl->GetAdapter(list, i, &IID_IDXCoreAdapter_, (void**)&a);
        if (FAILED(hr)) continue;

        bool is_hw = false;
        a->lpVtbl->GetProperty(a, DXCORE_ADAPTER_PROPERTY_IsHardware,
            sizeof(is_hw), &is_hw);

        if (is_hw) {
            adapter = a;
            /* Get adapter name */
            size_t name_size = 0;
            a->lpVtbl->GetPropertySize(a, DXCORE_ADAPTER_PROPERTY_DriverDescription, &name_size);
            if (name_size > 0 && name_size < sizeof(g.adapter_name)) {
                a->lpVtbl->GetProperty(a, DXCORE_ADAPTER_PROPERTY_DriverDescription,
                    name_size, g.adapter_name);
            }
            /* QI to IUnknown for D3D12CreateDevice */
            a->lpVtbl->QueryInterface(a, &IID_IUnknown, (void**)&adapter_unknown);
            break;
        }
        a->lpVtbl->Release(a);
    }

    list->lpVtbl->Release(list);
    factory->lpVtbl->Release(factory);

    if (!adapter) {
        set_error("No hardware D3D12 adapter found");
        return -1;
    }

    /* Create D3D12 device */
    hr = g.pfn_D3D12CreateDevice(adapter_unknown, D3D_FEATURE_LEVEL_11_0,
        &IID_ID3D12Device, (void**)&g.device);

    if (adapter_unknown) adapter_unknown->lpVtbl->Release(adapter_unknown);
    adapter->lpVtbl->Release(adapter);

    if (FAILED(hr)) {
        set_error("D3D12CreateDevice failed: 0x%08x", (unsigned)hr);
        return -1;
    }

    /* Create compute command queue */
    D3D12_COMMAND_QUEUE_DESC queue_desc = {0};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    queue_desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_HIGH;
    queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queue_desc.NodeMask = 0;

    hr = ID3D12Device_CreateCommandQueue(g.device, &queue_desc,
        &IID_ID3D12CommandQueue, (void**)&g.queue);
    if (FAILED(hr)) {
        set_error("CreateCommandQueue failed: 0x%08x", (unsigned)hr);
        return -1;
    }

    /* Create command allocator */
    hr = ID3D12Device_CreateCommandAllocator(g.device,
        D3D12_COMMAND_LIST_TYPE_COMPUTE,
        &IID_ID3D12CommandAllocator, (void**)&g.allocator);
    if (FAILED(hr)) {
        set_error("CreateCommandAllocator failed: 0x%08x", (unsigned)hr);
        return -1;
    }

    /* Create command list (closed initially) */
    hr = ID3D12Device_CreateCommandList(g.device, 0,
        D3D12_COMMAND_LIST_TYPE_COMPUTE, g.allocator, NULL,
        &IID_ID3D12GraphicsCommandList, (void**)&g.cmdlist);
    if (FAILED(hr)) {
        set_error("CreateCommandList failed: 0x%08x", (unsigned)hr);
        return -1;
    }
    /* Close immediately — we'll reset+reopen for each batch */
    ID3D12GraphicsCommandList_Close(g.cmdlist);

    /* Create fence */
    g.fence_value = 0;
    hr = ID3D12Device_CreateFence(g.device, 0, D3D12_FENCE_FLAG_NONE,
        &IID_ID3D12Fence, (void**)&g.fence);
    if (FAILED(hr)) {
        set_error("CreateFence failed: 0x%08x", (unsigned)hr);
        return -1;
    }

    /* Create shader-visible CBV/SRV/UAV descriptor heap */
    {
        D3D12_DESCRIPTOR_HEAP_DESC heap_desc;
        memset(&heap_desc, 0, sizeof(heap_desc));
        heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heap_desc.NumDescriptors = DESC_HEAP_SIZE;
        heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        heap_desc.NodeMask = 0;

        hr = ID3D12Device_CreateDescriptorHeap(g.device, &heap_desc,
            &IID_ID3D12DescriptorHeap, (void**)&g.desc_heap);
        if (FAILED(hr)) {
            set_error("CreateDescriptorHeap failed: 0x%08x", (unsigned)hr);
            return -1;
        }

        g.desc_size = ID3D12Device_GetDescriptorHandleIncrementSize(
            g.device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        g.desc_heap_offset = 0;
    }

    g.next_handle = 0;
    g.initialized = true;

    /* Check feature support */
    {
        /* D3D12_FEATURE_D3D12_OPTIONS = 0 */
        struct { BOOL DoublePrecisionFloatShaderOps; BOOL OutputMergerLogicOp;
                 uint32_t MinPrecisionSupport; uint32_t TiledResourcesTier;
                 uint32_t ResourceBindingTier; BOOL PSSpecifiedStencilRefSupported;
                 BOOL TypedUAVLoadAdditionalFormats; BOOL ROVsSupported;
                 uint32_t ConservativeRasterizationTier; BOOL StandardSwizzle64KBSupported;
                 BOOL CrossAdapterRowMajorTextureSupported; BOOL VPAndRTArrayIndexFromAnyShaderFeedingRasterizerSupportedWithoutGSEmulation;
                 uint32_t ResourceHeapTier; } opts;
        memset(&opts, 0, sizeof(opts));
        hr = ID3D12Device_CheckFeatureSupport(g.device, 0 /*D3D12_FEATURE_D3D12_OPTIONS*/, &opts, sizeof(opts));
        if (SUCCEEDED(hr)) {
            fprintf(stderr, "[d3d12c] ResourceBindingTier=%u ResourceHeapTier=%u TypedUAVLoad=%d\n",
                opts.ResourceBindingTier, opts.ResourceHeapTier, opts.TypedUAVLoadAdditionalFormats);
        }

        /* D3D12_FEATURE_SHADER_MODEL = 7 */
        struct { uint32_t HighestShaderModel; } sm;
        sm.HighestShaderModel = SHADER_MODEL_6_0_;
        hr = ID3D12Device_CheckFeatureSupport(g.device, 7 /*D3D12_FEATURE_SHADER_MODEL*/, &sm, sizeof(sm));
        if (SUCCEEDED(hr)) {
            fprintf(stderr, "[d3d12c] HighestShaderModel=0x%x (6.%d)\n",
                sm.HighestShaderModel, sm.HighestShaderModel & 0xf);
        }

        /* D3D12_FEATURE_ARCHITECTURE1 = 16 */
        struct { UINT NodeIndex; BOOL TileBasedRenderer; BOOL UMA; BOOL CacheCoherentUMA; BOOL IsolatedMMU; } arch;
        memset(&arch, 0, sizeof(arch));
        hr = ID3D12Device_CheckFeatureSupport(g.device, 16 /*D3D12_FEATURE_ARCHITECTURE1*/, &arch, sizeof(arch));
        if (SUCCEEDED(hr)) {
            fprintf(stderr, "[d3d12c] TileBasedRenderer=%d UMA=%d\n", arch.TileBasedRenderer, arch.UMA);
        }
    }

    fprintf(stderr, "[d3d12c] Initialized: %s\n",
        g.adapter_name[0] ? g.adapter_name : "Unknown GPU");
    return 0;
}

/* ============================================================================
 * Public API: Shutdown
 * ============================================================================ */

void d3d12c_shutdown(void) {
    if (!g.initialized) return;

    /* Release all handles */
    for (uint32_t i = 0; i < MAX_HANDLES; i++) {
        HandleEntry *e = &g.handles[i];
        if (e->type == HANDLE_BUFFER && e->resource) {
            ID3D12Resource_Release(e->resource);
        } else if (e->type == HANDLE_PIPELINE) {
            if (e->pipeline.pso) ID3D12PipelineState_Release(e->pipeline.pso);
            if (e->pipeline.root_sig) ID3D12RootSignature_Release(e->pipeline.root_sig);
        }
    }

    if (g.desc_heap) ID3D12DescriptorHeap_Release(g.desc_heap);
    if (g.fence) ID3D12Fence_Release(g.fence);
    if (g.cmdlist) ID3D12GraphicsCommandList_Release(g.cmdlist);
    if (g.allocator) ID3D12CommandAllocator_Release(g.allocator);
    if (g.queue) ID3D12CommandQueue_Release(g.queue);
    if (g.device) ID3D12Device_Release(g.device);

    if (g.lib_spirv2dxil) dlclose(g.lib_spirv2dxil);
    if (g.lib_dxcore) dlclose(g.lib_dxcore);
    if (g.lib_d3d12) dlclose(g.lib_d3d12);

    memset(&g, 0, sizeof(g));
}

/* ============================================================================
 * Public API: Device info
 * ============================================================================ */

const char* d3d12c_get_adapter_name(void) {
    return g.adapter_name;
}

const char* d3d12c_get_last_error(void) {
    return g_last_error;
}

/* ============================================================================
 * Public API: Buffer management
 * ============================================================================ */

static uint64_t create_buffer_internal(uint64_t size_bytes, D3D12_HEAP_TYPE heap_type) {
    if (!g.initialized) return 0;

    /* Align to 256 bytes (D3D12 requirement for constant buffers) */
    uint64_t aligned_size = (size_bytes + 255) & ~255ULL;
    if (aligned_size == 0) aligned_size = 256;

    D3D12_HEAP_PROPERTIES heap_props = {0};
    heap_props.Type = heap_type;
    heap_props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

    D3D12_RESOURCE_DESC desc = {0};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = aligned_size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = (heap_type == D3D12_HEAP_TYPE_DEFAULT)
        ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        : D3D12_RESOURCE_FLAG_NONE;

    D3D12_RESOURCE_STATES initial_state;
    if (heap_type == D3D12_HEAP_TYPE_UPLOAD)
        initial_state = D3D12_RESOURCE_STATE_GENERIC_READ;
    else if (heap_type == D3D12_HEAP_TYPE_READBACK)
        initial_state = D3D12_RESOURCE_STATE_COPY_DEST;
    else
        initial_state = D3D12_RESOURCE_STATE_COMMON;

    ID3D12Resource *resource = NULL;
    HRESULT hr = ID3D12Device_CreateCommittedResource(g.device,
        &heap_props, D3D12_HEAP_FLAG_NONE, &desc, initial_state,
        NULL, &IID_ID3D12Resource, (void**)&resource);

    if (FAILED(hr)) {
        set_error("CreateCommittedResource failed: 0x%08x (size=%llu, heap=%d)",
            (unsigned)hr, (unsigned long long)size_bytes, heap_type);
        return 0;
    }

    uint64_t h = alloc_handle();
    if (h == 0) {
        ID3D12Resource_Release(resource);
        set_error("Handle table full");
        return 0;
    }

    g.handles[h - 1].type = HANDLE_BUFFER;
    g.handles[h - 1].resource = resource;
    return h;
}

uint64_t d3d12c_create_buffer(uint64_t size_bytes) {
    return create_buffer_internal(size_bytes, D3D12_HEAP_TYPE_DEFAULT);
}

uint64_t d3d12c_create_upload_buffer(uint64_t size_bytes) {
    return create_buffer_internal(size_bytes, D3D12_HEAP_TYPE_UPLOAD);
}

uint64_t d3d12c_create_readback_buffer(uint64_t size_bytes) {
    return create_buffer_internal(size_bytes, D3D12_HEAP_TYPE_READBACK);
}

void d3d12c_release_buffer(uint64_t handle) {
    HandleEntry *e = get_handle(handle);
    if (!e || e->type != HANDLE_BUFFER) return;
    if (e->resource) ID3D12Resource_Release(e->resource);
    e->type = HANDLE_NONE;
    e->resource = NULL;
}

/* ============================================================================
 * Public API: Data transfer (CPU ↔ GPU)
 * ============================================================================ */

int d3d12c_upload(uint64_t dst_handle, const void *data, uint64_t size_bytes) {
    if (!g.initialized) return -1;
    HandleEntry *dst = get_handle(dst_handle);
    if (!dst || dst->type != HANDLE_BUFFER) {
        set_error("upload: invalid dst handle %llu", (unsigned long long)dst_handle);
        return -1;
    }

    /* Create temporary upload buffer */
    uint64_t upload_h = d3d12c_create_upload_buffer(size_bytes);
    if (upload_h == 0) return -1;
    HandleEntry *upload = get_handle(upload_h);

    /* Map, copy, unmap */
    void *mapped = NULL;
    D3D12_RANGE read_range = {0, 0}; /* We won't read from this */
    HRESULT hr = ID3D12Resource_Map(upload->resource, 0, &read_range, &mapped);
    if (FAILED(hr)) {
        d3d12c_release_buffer(upload_h);
        set_error("Map failed: 0x%08x", (unsigned)hr);
        return -1;
    }
    memcpy(mapped, data, size_bytes);
    ID3D12Resource_Unmap(upload->resource, 0, NULL);

    /* Record copy command */
    ID3D12CommandAllocator_Reset(g.allocator);
    ID3D12GraphicsCommandList_Reset(g.cmdlist, g.allocator, NULL);

    ID3D12GraphicsCommandList_CopyBufferRegion(g.cmdlist,
        dst->resource, 0, upload->resource, 0, size_bytes);

    /* UAV barrier so the buffer is ready for compute */
    D3D12_RESOURCE_BARRIER barrier = {0};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = dst->resource;
    ID3D12GraphicsCommandList_ResourceBarrier(g.cmdlist, 1, &barrier);

    ID3D12GraphicsCommandList_Close(g.cmdlist);

    ID3D12CommandList *lists[] = { (ID3D12CommandList*)g.cmdlist };
    ID3D12CommandQueue_ExecuteCommandLists(g.queue, 1, lists);
    wait_for_gpu();

    d3d12c_release_buffer(upload_h);
    return 0;
}

int d3d12c_readback(uint64_t src_handle, void *data, uint64_t size_bytes) {
    if (!g.initialized) return -1;
    HandleEntry *src = get_handle(src_handle);
    if (!src || src->type != HANDLE_BUFFER) {
        set_error("readback: invalid src handle %llu", (unsigned long long)src_handle);
        return -1;
    }

    /* Create temporary readback buffer */
    uint64_t rb_h = d3d12c_create_readback_buffer(size_bytes);
    if (rb_h == 0) return -1;
    HandleEntry *rb = get_handle(rb_h);

    /* Record copy command */
    ID3D12CommandAllocator_Reset(g.allocator);
    ID3D12GraphicsCommandList_Reset(g.cmdlist, g.allocator, NULL);

    ID3D12GraphicsCommandList_CopyBufferRegion(g.cmdlist,
        rb->resource, 0, src->resource, 0, size_bytes);

    ID3D12GraphicsCommandList_Close(g.cmdlist);

    ID3D12CommandList *lists[] = { (ID3D12CommandList*)g.cmdlist };
    ID3D12CommandQueue_ExecuteCommandLists(g.queue, 1, lists);
    wait_for_gpu();

    /* Map readback buffer and copy to CPU */
    void *mapped = NULL;
    D3D12_RANGE read_range = {0, size_bytes};
    HRESULT hr = ID3D12Resource_Map(rb->resource, 0, &read_range, &mapped);
    if (FAILED(hr)) {
        d3d12c_release_buffer(rb_h);
        set_error("Map readback failed: 0x%08x", (unsigned)hr);
        return -1;
    }
    memcpy(data, mapped, size_bytes);
    D3D12_RANGE written_range = {0, 0};
    ID3D12Resource_Unmap(rb->resource, 0, &written_range);

    d3d12c_release_buffer(rb_h);
    return 0;
}

/* ============================================================================
 * DXBC/PSV0 resource binding parser
 *
 * spirv_to_dxil generates DXIL with specific resource types based on SPIR-V
 * decorations. NonWritable SSBOs become SRVs, writable ones become UAVs.
 * The root signature MUST match these bindings exactly.
 * ============================================================================ */

/* PSV resource types (from DXC DxilPipelineStateValidation.h) */
#define PSV_INVALID     0
#define PSV_SAMPLER     1
#define PSV_CBV         2
#define PSV_SRV_TYPED   3
#define PSV_SRV_RAW     4
#define PSV_SRV_STRUCT  5
#define PSV_UAV_TYPED   6
#define PSV_UAV_RAW     7
#define PSV_UAV_STRUCT  8
#define PSV_UAV_STRUCT_CTR 9

typedef struct {
    uint32_t num_srvs;
    uint32_t num_uavs;
    uint32_t num_cbvs;
    uint32_t num_samplers;
    /* Track register ranges for root signature */
    uint32_t srv_min_reg, srv_max_reg;
    uint32_t uav_min_reg, uav_max_reg;
    uint32_t cbv_min_reg, cbv_max_reg;
} dxil_resource_info;

static bool parse_dxbc_resources(const void *dxbc, size_t dxbc_size,
                                  dxil_resource_info *info)
{
    memset(info, 0, sizeof(*info));
    info->srv_min_reg = UINT32_MAX;
    info->uav_min_reg = UINT32_MAX;
    info->cbv_min_reg = UINT32_MAX;

    const uint8_t *p = (const uint8_t *)dxbc;
    if (dxbc_size < 32 || memcmp(p, "DXBC", 4) != 0) return false;

    uint32_t num_parts;
    memcpy(&num_parts, p + 28, 4);

    for (uint32_t i = 0; i < num_parts; i++) {
        uint32_t offset;
        memcpy(&offset, p + 32 + i * 4, 4);
        if (offset + 8 > dxbc_size) continue;

        if (memcmp(p + offset, "PSV0", 4) != 0) continue;

        uint32_t part_size;
        memcpy(&part_size, p + offset + 4, 4);
        const uint8_t *psv = p + offset + 8;
        if (offset + 8 + part_size > dxbc_size) return false;

        /* PSV0 header: first 4 bytes = header size */
        uint32_t hdr_size;
        memcpy(&hdr_size, psv, 4);

        /* Resource count is at header_size + 0 in the post-header area */
        uint32_t post_hdr = hdr_size + 4; /* +4 for header size field itself */
        if (post_hdr + 8 > part_size) return true; /* No resources */

        uint32_t num_resources, resource_stride;
        memcpy(&num_resources, psv + post_hdr, 4);
        memcpy(&resource_stride, psv + post_hdr + 4, 4);

        uint32_t res_start = post_hdr + 8;
        for (uint32_t r = 0; r < num_resources; r++) {
            uint32_t res_off = res_start + r * resource_stride;
            if (res_off + 16 > part_size) break;

            uint32_t res_type, space, lower_bound, upper_bound;
            memcpy(&res_type, psv + res_off, 4);
            memcpy(&space, psv + res_off + 4, 4);
            memcpy(&lower_bound, psv + res_off + 8, 4);
            memcpy(&upper_bound, psv + res_off + 12, 4);

            uint32_t count = upper_bound - lower_bound + 1;

            if (res_type >= PSV_SRV_TYPED && res_type <= PSV_SRV_STRUCT) {
                info->num_srvs += count;
                if (lower_bound < info->srv_min_reg) info->srv_min_reg = lower_bound;
                if (upper_bound > info->srv_max_reg) info->srv_max_reg = upper_bound;
            } else if (res_type >= PSV_UAV_TYPED && res_type <= PSV_UAV_STRUCT_CTR) {
                info->num_uavs += count;
                if (lower_bound < info->uav_min_reg) info->uav_min_reg = lower_bound;
                if (upper_bound > info->uav_max_reg) info->uav_max_reg = upper_bound;
            } else if (res_type == PSV_CBV) {
                info->num_cbvs += count;
                if (lower_bound < info->cbv_min_reg) info->cbv_min_reg = lower_bound;
                if (upper_bound > info->cbv_max_reg) info->cbv_max_reg = upper_bound;
            } else if (res_type == PSV_SAMPLER) {
                info->num_samplers += count;
            }
        }

        fprintf(stderr, "[d3d12c] PSV0: %u resources (%u SRVs t%u-%u, %u UAVs u%u-%u, %u CBVs b%u-%u)\n",
            num_resources, info->num_srvs,
            info->srv_min_reg == UINT32_MAX ? 0 : info->srv_min_reg,
            info->num_srvs ? info->srv_max_reg : 0,
            info->num_uavs,
            info->uav_min_reg == UINT32_MAX ? 0 : info->uav_min_reg,
            info->num_uavs ? info->uav_max_reg : 0,
            info->num_cbvs,
            info->cbv_min_reg == UINT32_MAX ? 0 : info->cbv_min_reg,
            info->num_cbvs ? info->cbv_max_reg : 0);
        return true;
    }
    return false; /* No PSV0 found */
}

/* Build root signature from DXIL resource bindings */
static HRESULT create_root_sig_from_dxil(const void *dxil, size_t dxil_size,
                                          ID3D12RootSignature **out_root_sig,
                                          dxil_resource_info *out_info)
{
    dxil_resource_info info;
    parse_dxbc_resources(dxil, dxil_size, &info);
    if (out_info) *out_info = info;

    uint32_t num_params = 0;
    D3D12_ROOT_PARAMETER params[4]; /* SRV, UAV, CBV, Sampler tables */
    D3D12_DESCRIPTOR_RANGE ranges[4];
    memset(params, 0, sizeof(params));
    memset(ranges, 0, sizeof(ranges));

    if (info.num_srvs > 0) {
        uint32_t base = info.srv_min_reg;
        uint32_t count = info.srv_max_reg - base + 1;
        ranges[num_params].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        ranges[num_params].NumDescriptors = count;
        ranges[num_params].BaseShaderRegister = base;
        ranges[num_params].RegisterSpace = 0;
        ranges[num_params].OffsetInDescriptorsFromTableStart = 0;
        params[num_params].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[num_params].DescriptorTable.NumDescriptorRanges = 1;
        params[num_params].DescriptorTable.pDescriptorRanges = &ranges[num_params];
        params[num_params].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        num_params++;
    }

    if (info.num_uavs > 0) {
        uint32_t base = info.uav_min_reg;
        uint32_t count = info.uav_max_reg - base + 1;
        ranges[num_params].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        ranges[num_params].NumDescriptors = count;
        ranges[num_params].BaseShaderRegister = base;
        ranges[num_params].RegisterSpace = 0;
        ranges[num_params].OffsetInDescriptorsFromTableStart = 0;
        params[num_params].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[num_params].DescriptorTable.NumDescriptorRanges = 1;
        params[num_params].DescriptorTable.pDescriptorRanges = &ranges[num_params];
        params[num_params].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        num_params++;
    }

    if (info.num_cbvs > 0) {
        uint32_t base = info.cbv_min_reg;
        uint32_t count = info.cbv_max_reg - base + 1;
        ranges[num_params].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
        ranges[num_params].NumDescriptors = count;
        ranges[num_params].BaseShaderRegister = base;
        ranges[num_params].RegisterSpace = 0;
        ranges[num_params].OffsetInDescriptorsFromTableStart = 0;
        params[num_params].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[num_params].DescriptorTable.NumDescriptorRanges = 1;
        params[num_params].DescriptorTable.pDescriptorRanges = &ranges[num_params];
        params[num_params].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        num_params++;
    }

    D3D12_ROOT_SIGNATURE_DESC rs_desc;
    memset(&rs_desc, 0, sizeof(rs_desc));
    rs_desc.NumParameters = num_params;
    rs_desc.pParameters = params;
    rs_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ID3D10Blob *rs_blob = NULL;
    ID3D10Blob *error_blob = NULL;
    HRESULT hr = g.pfn_D3D12SerializeRootSignature(&rs_desc,
        D3D_ROOT_SIGNATURE_VERSION_1_0, &rs_blob, &error_blob);

    if (FAILED(hr)) {
        if (error_blob) {
            fprintf(stderr, "[d3d12c] SerializeRootSignature: %s\n",
                (char*)ID3D10Blob_GetBufferPointer(error_blob));
            ID3D10Blob_Release(error_blob);
        }
        return hr;
    }

    hr = ID3D12Device_CreateRootSignature(g.device, 0,
        ID3D10Blob_GetBufferPointer(rs_blob),
        ID3D10Blob_GetBufferSize(rs_blob),
        &IID_ID3D12RootSignature, (void**)out_root_sig);
    ID3D10Blob_Release(rs_blob);
    return hr;
}

/* Logger callback for spirv_to_dxil (safe non-variadic wrapper) */
static void spirv_log_cb(void *priv, const char *msg) {
    fprintf(stderr, "[spirv2dxil] %s\n", msg);
}

/* ============================================================================
 * Public API: Shader compilation (SPIR-V → DXIL → PSO)
 * ============================================================================ */

uint64_t d3d12c_create_compute_pipeline(
    const void *spirv_data, uint32_t spirv_size,
    uint32_t num_uavs, uint32_t num_cbvs)
{
    if (!g.initialized) return 0;

    /* Convert byte count to word count (SPIR-V is 32-bit aligned) */
    const uint32_t *spirv = (const uint32_t *)spirv_data;
    uint32_t spirv_words = spirv_size / 4;

    /* SPIR-V → DXIL via Mesa's spirv_to_dxil */
    struct spirv_to_dxil_debug_options debug_opts = {0};
    /* Enable NIR dump via env var for debugging */
    if (getenv("D3D12C_DUMP_NIR"))
        debug_opts.dump_nir = true;
    struct spirv_to_dxil_runtime_conf conf = {0};
    conf.shader_model_max = SHADER_MODEL_6_0_;
    struct spirv_to_dxil_logger logger = {0};
    logger.priv = stderr;
    logger.log = spirv_log_cb;
    struct spirv_to_dxil_object dxil_obj = {0};

    fprintf(stderr, "[d3d12c] Compiling SPIR-V (%u bytes, %u words) to DXIL...\n",
        spirv_size, spirv_words);
    bool ok = g.pfn_spirv_to_dxil(spirv, spirv_words,
        NULL, 0,
        DXIL_SPIRV_SHADER_COMPUTE_, "main",
        NO_DXIL_VALIDATION_,
        &debug_opts, &conf, &logger, &dxil_obj);

    if (!ok || !dxil_obj.binary.buffer) {
        set_error("spirv_to_dxil compilation failed");
        return 0;
    }
    fprintf(stderr, "[d3d12c] DXIL generated: %zu bytes\n", dxil_obj.binary.size);

    /* Build root signature from the DXIL's PSV0 resource bindings.
     * spirv_to_dxil may convert NonWritable SSBOs to SRVs, so we can't just
     * assume everything is a UAV. Parse the DXBC to get exact resource layout. */
    ID3D12RootSignature *root_sig = NULL;
    dxil_resource_info res_info;
    HRESULT hr = create_root_sig_from_dxil(dxil_obj.binary.buffer,
        dxil_obj.binary.size, &root_sig, &res_info);

    if (FAILED(hr)) {
        set_error("CreateRootSignature from DXIL failed: 0x%08x", (unsigned)hr);
        g.pfn_spirv_to_dxil_free(&dxil_obj);
        return 0;
    }

    /* Create compute pipeline state — try Device2 stream API first (Dozen path),
     * fall back to legacy CreateComputePipelineState */
    ID3D12PipelineState *pso = NULL;

    /* Try Device4::CreatePipelineState (stream API — same as Dozen driver) */
    ID3D12Device4 *device4 = NULL;
    hr = ID3D12Device_QueryInterface(g.device, &IID_ID3D12Device4, (void**)&device4);
    if (SUCCEEDED(hr) && device4) {
        d3d12_compute_stream stream;
        memset(&stream, 0, sizeof(stream));
        stream.root_sig.type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE;
        stream.root_sig.root_sig = root_sig;
        stream.cs.type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS;
        stream.cs.cs.pShaderBytecode = dxil_obj.binary.buffer;
        stream.cs.cs.BytecodeLength = dxil_obj.binary.size;

        D3D12_PIPELINE_STATE_STREAM_DESC stream_desc;
        stream_desc.SizeInBytes = sizeof(stream);
        stream_desc.pPipelineStateSubobjectStream = &stream;

        fprintf(stderr, "[d3d12c] Trying Device4 stream API (sizeof stream=%zu)\n", sizeof(stream));
        hr = ID3D12Device4_CreatePipelineState(device4, &stream_desc,
            &IID_ID3D12PipelineState, (void**)&pso);
        ID3D12Device4_Release(device4);

        if (SUCCEEDED(hr)) {
            fprintf(stderr, "[d3d12c] Pipeline created via Device4 stream API\n");
        } else {
            fprintf(stderr, "[d3d12c] Device4 stream API failed: 0x%08x, trying legacy\n",
                (unsigned)hr);
            pso = NULL;  /* Ensure fallback path triggers */
        }
    } else {
        fprintf(stderr, "[d3d12c] Device4 not available (hr=0x%08x)\n", (unsigned)hr);
    }

    /* Fallback: legacy CreateComputePipelineState */
    if (!pso) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc = {0};
        pso_desc.pRootSignature = root_sig;
        pso_desc.CS.pShaderBytecode = dxil_obj.binary.buffer;
        pso_desc.CS.BytecodeLength = dxil_obj.binary.size;

        hr = ID3D12Device_CreateComputePipelineState(g.device, &pso_desc,
            &IID_ID3D12PipelineState, (void**)&pso);
    }

    g.pfn_spirv_to_dxil_free(&dxil_obj);

    if (FAILED(hr)) {
        ID3D12RootSignature_Release(root_sig);
        set_error("CreateComputePipelineState failed: 0x%08x", (unsigned)hr);
        return 0;
    }

    /* Store in handle table */
    uint64_t h = alloc_handle();
    if (h == 0) {
        ID3D12PipelineState_Release(pso);
        ID3D12RootSignature_Release(root_sig);
        set_error("Handle table full");
        return 0;
    }

    g.handles[h - 1].type = HANDLE_PIPELINE;
    g.handles[h - 1].pipeline.pso = pso;
    g.handles[h - 1].pipeline.root_sig = root_sig;
    g.handles[h - 1].pipeline.num_srvs = res_info.num_srvs;
    g.handles[h - 1].pipeline.num_uavs = res_info.num_uavs;
    g.handles[h - 1].pipeline.num_cbvs = res_info.num_cbvs;
    return h;
}

/* Create pipeline from pre-compiled DXIL bytecode (skip SPIR-V step) */
uint64_t d3d12c_create_pipeline_from_dxil(
    const void *dxil, uint32_t dxil_size,
    uint32_t num_uavs, uint32_t num_cbvs)
{
    if (!g.initialized) return 0;

    /* Build root signature from DXIL PSV0 resource bindings */
    ID3D12RootSignature *root_sig = NULL;
    dxil_resource_info res_info;
    HRESULT hr = create_root_sig_from_dxil(dxil, dxil_size, &root_sig, &res_info);

    if (FAILED(hr)) {
        set_error("CreateRootSignature from DXIL failed: 0x%08x", (unsigned)hr);
        return 0;
    }

    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc = {0};
    pso_desc.pRootSignature = root_sig;
    pso_desc.CS.pShaderBytecode = dxil;
    pso_desc.CS.BytecodeLength = dxil_size;

    ID3D12PipelineState *pso = NULL;
    hr = ID3D12Device_CreateComputePipelineState(g.device, &pso_desc,
        &IID_ID3D12PipelineState, (void**)&pso);

    if (FAILED(hr)) {
        ID3D12RootSignature_Release(root_sig);
        set_error("CreateComputePipelineState failed: 0x%08x", (unsigned)hr);
        return 0;
    }

    uint64_t h = alloc_handle();
    if (h == 0) {
        ID3D12PipelineState_Release(pso);
        ID3D12RootSignature_Release(root_sig);
        return 0;
    }

    g.handles[h - 1].type = HANDLE_PIPELINE;
    g.handles[h - 1].pipeline.pso = pso;
    g.handles[h - 1].pipeline.root_sig = root_sig;
    g.handles[h - 1].pipeline.num_srvs = res_info.num_srvs;
    g.handles[h - 1].pipeline.num_uavs = res_info.num_uavs;
    g.handles[h - 1].pipeline.num_cbvs = res_info.num_cbvs;
    return h;
}

void d3d12c_release_pipeline(uint64_t handle) {
    HandleEntry *e = get_handle(handle);
    if (!e || e->type != HANDLE_PIPELINE) return;
    if (e->pipeline.pso) ID3D12PipelineState_Release(e->pipeline.pso);
    if (e->pipeline.root_sig) ID3D12RootSignature_Release(e->pipeline.root_sig);
    e->type = HANDLE_NONE;
}

/* ============================================================================
 * Public API: Dispatch
 * ============================================================================ */

int d3d12c_begin_commands(void) {
    if (!g.initialized) return -1;
    HRESULT hr = ID3D12CommandAllocator_Reset(g.allocator);
    if (FAILED(hr)) { set_error("Reset allocator: 0x%08x", (unsigned)hr); return -1; }
    hr = ID3D12GraphicsCommandList_Reset(g.cmdlist, g.allocator, NULL);
    if (FAILED(hr)) { set_error("Reset cmdlist: 0x%08x", (unsigned)hr); return -1; }
    /* Reset descriptor heap offset for new batch */
    g.desc_heap_offset = 0;
    return 0;
}

int d3d12c_dispatch(
    uint64_t pipeline_handle,
    const uint64_t *srv_handles, uint32_t num_srvs,
    const uint64_t *uav_handles, uint32_t num_uavs,
    const uint64_t *cbv_handles, uint32_t num_cbvs,
    uint32_t groups_x, uint32_t groups_y, uint32_t groups_z)
{
    if (!g.initialized) return -1;

    HandleEntry *pe = get_handle(pipeline_handle);
    if (!pe || pe->type != HANDLE_PIPELINE) {
        set_error("dispatch: invalid pipeline handle");
        return -1;
    }

    /* Set descriptor heap */
    ID3D12DescriptorHeap *heaps[] = { g.desc_heap };
    ID3D12GraphicsCommandList_SetDescriptorHeaps(g.cmdlist, 1, heaps);

    ID3D12GraphicsCommandList_SetComputeRootSignature(g.cmdlist, pe->pipeline.root_sig);
    ID3D12GraphicsCommandList_SetPipelineState(g.cmdlist, pe->pipeline.pso);

    /* Allocate descriptors from heap for this dispatch */
    uint32_t total_descs = num_srvs + num_uavs + num_cbvs;
    uint32_t base = g.desc_heap_offset;
    if (base + total_descs > DESC_HEAP_SIZE) {
        set_error("dispatch: descriptor heap exhausted");
        return -1;
    }
    g.desc_heap_offset += total_descs;

    /* Get heap start handles */
    D3D12_CPU_DESCRIPTOR_HANDLE cpu_start =
        ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(g.desc_heap);
    D3D12_GPU_DESCRIPTOR_HANDLE gpu_start =
        ID3D12DescriptorHeap_GetGPUDescriptorHandleForHeapStart(g.desc_heap);

    /* Root parameter order matches create_root_sig_from_dxil:
     * SRV table (if any) → UAV table (if any) → CBV table (if any) */
    uint32_t root_param_idx = 0;
    uint32_t desc_offset = base;

    /* Create SRV descriptors (raw buffer views for read-only SSBOs) */
    if (num_srvs > 0) {
        for (uint32_t i = 0; i < num_srvs; i++) {
            HandleEntry *be = get_handle(srv_handles[i]);
            if (!be || be->type != HANDLE_BUFFER) {
                set_error("dispatch: invalid SRV handle at index %u", i);
                return -1;
            }

            D3D12_RESOURCE_DESC rdesc = ID3D12Resource_GetDesc(be->resource);

            D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc;
            memset(&srv_desc, 0, sizeof(srv_desc));
            srv_desc.Format = DXGI_FORMAT_R32_TYPELESS;
            srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv_desc.Buffer.FirstElement = 0;
            srv_desc.Buffer.NumElements = (UINT)(rdesc.Width / 4);
            srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;

            D3D12_CPU_DESCRIPTOR_HANDLE dest;
            dest.ptr = cpu_start.ptr + (SIZE_T)(desc_offset + i) * g.desc_size;
            ID3D12Device_CreateShaderResourceView(g.device, be->resource,
                &srv_desc, dest);
        }

        D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle;
        gpu_handle.ptr = gpu_start.ptr + (UINT64)desc_offset * g.desc_size;
        ID3D12GraphicsCommandList_SetComputeRootDescriptorTable(
            g.cmdlist, root_param_idx, gpu_handle);
        root_param_idx++;
        desc_offset += num_srvs;
    }

    /* Create UAV descriptors (raw buffer views for read-write SSBOs) */
    if (num_uavs > 0) {
        for (uint32_t i = 0; i < num_uavs; i++) {
            HandleEntry *be = get_handle(uav_handles[i]);
            if (!be || be->type != HANDLE_BUFFER) {
                set_error("dispatch: invalid UAV handle at index %u", i);
                return -1;
            }

            D3D12_RESOURCE_DESC rdesc = ID3D12Resource_GetDesc(be->resource);

            D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc;
            memset(&uav_desc, 0, sizeof(uav_desc));
            uav_desc.Format = DXGI_FORMAT_R32_TYPELESS;
            uav_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            uav_desc.Buffer.FirstElement = 0;
            uav_desc.Buffer.NumElements = (UINT)(rdesc.Width / 4);
            uav_desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;

            D3D12_CPU_DESCRIPTOR_HANDLE dest;
            dest.ptr = cpu_start.ptr + (SIZE_T)(desc_offset + i) * g.desc_size;
            ID3D12Device_CreateUnorderedAccessView(g.device, be->resource,
                NULL, &uav_desc, dest);
        }

        D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle;
        gpu_handle.ptr = gpu_start.ptr + (UINT64)desc_offset * g.desc_size;
        ID3D12GraphicsCommandList_SetComputeRootDescriptorTable(
            g.cmdlist, root_param_idx, gpu_handle);
        root_param_idx++;
        desc_offset += num_uavs;
    }

    /* Create CBV descriptors */
    if (num_cbvs > 0) {
        for (uint32_t i = 0; i < num_cbvs; i++) {
            HandleEntry *be = get_handle(cbv_handles[i]);
            if (!be || be->type != HANDLE_BUFFER) {
                set_error("dispatch: invalid CBV handle at index %u", i);
                return -1;
            }

            D3D12_RESOURCE_DESC rdesc = ID3D12Resource_GetDesc(be->resource);

            D3D12_CONSTANT_BUFFER_VIEW_DESC cbv_desc;
            memset(&cbv_desc, 0, sizeof(cbv_desc));
            cbv_desc.BufferLocation = ID3D12Resource_GetGPUVirtualAddress(be->resource);
            cbv_desc.SizeInBytes = (UINT)((rdesc.Width + 255) & ~255ULL);

            D3D12_CPU_DESCRIPTOR_HANDLE dest;
            dest.ptr = cpu_start.ptr + (SIZE_T)(desc_offset + i) * g.desc_size;
            ID3D12Device_CreateConstantBufferView(g.device, &cbv_desc, dest);
        }

        D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle;
        gpu_handle.ptr = gpu_start.ptr + (UINT64)desc_offset * g.desc_size;
        ID3D12GraphicsCommandList_SetComputeRootDescriptorTable(
            g.cmdlist, root_param_idx, gpu_handle);
        root_param_idx++;
    }

    ID3D12GraphicsCommandList_Dispatch(g.cmdlist, groups_x, groups_y, groups_z);

    /* UAV barrier for synchronization between dispatches */
    D3D12_RESOURCE_BARRIER barrier = {0};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = NULL; /* All UAVs */
    ID3D12GraphicsCommandList_ResourceBarrier(g.cmdlist, 1, &barrier);

    return 0;
}

int d3d12c_end_commands_and_wait(void) {
    if (!g.initialized) return -1;

    HRESULT hr = ID3D12GraphicsCommandList_Close(g.cmdlist);
    if (FAILED(hr)) { set_error("Close cmdlist: 0x%08x", (unsigned)hr); return -1; }

    ID3D12CommandList *lists[] = { (ID3D12CommandList*)g.cmdlist };
    ID3D12CommandQueue_ExecuteCommandLists(g.queue, 1, lists);

    hr = wait_for_gpu();
    if (FAILED(hr)) { set_error("wait_for_gpu: 0x%08x", (unsigned)hr); return -1; }

    return 0;
}

/* Convenience: begin + single dispatch + end+wait */
int d3d12c_dispatch_sync(
    uint64_t pipeline_handle,
    const uint64_t *srv_handles, uint32_t num_srvs,
    const uint64_t *uav_handles, uint32_t num_uavs,
    const uint64_t *cbv_handles, uint32_t num_cbvs,
    uint32_t groups_x, uint32_t groups_y, uint32_t groups_z)
{
    if (d3d12c_begin_commands() != 0) return -1;
    if (d3d12c_dispatch(pipeline_handle, srv_handles, num_srvs,
            uav_handles, num_uavs,
            cbv_handles, num_cbvs, groups_x, groups_y, groups_z) != 0) return -1;
    return d3d12c_end_commands_and_wait();
}

/* ============================================================================
 * Public API: SPIR-V compilation helper
 * ============================================================================ */

int d3d12c_compile_spirv_to_dxil(
    const void *spirv_data, uint32_t spirv_size,
    void **out_dxil, uint32_t *out_dxil_size)
{
    if (!g.pfn_spirv_to_dxil) {
        set_error("spirv_to_dxil not loaded");
        return -1;
    }

    const uint32_t *spirv = (const uint32_t *)spirv_data;
    uint32_t spirv_words = spirv_size / 4;

    struct spirv_to_dxil_debug_options debug_opts = {0};
    struct spirv_to_dxil_runtime_conf conf = {0};
    conf.shader_model_max = SHADER_MODEL_6_0_;
    struct spirv_to_dxil_logger logger = {0};
    struct spirv_to_dxil_object dxil_obj = {0};

    bool ok = g.pfn_spirv_to_dxil(spirv, spirv_words,
        NULL, 0,
        DXIL_SPIRV_SHADER_COMPUTE_, "main",
        NO_DXIL_VALIDATION_,
        &debug_opts, &conf, &logger, &dxil_obj);

    if (!ok || !dxil_obj.binary.buffer) {
        set_error("SPIR-V → DXIL compilation failed");
        return -1;
    }

    /* Copy output (caller must free) */
    *out_dxil = malloc(dxil_obj.binary.size);
    memcpy(*out_dxil, dxil_obj.binary.buffer, dxil_obj.binary.size);
    *out_dxil_size = (uint32_t)dxil_obj.binary.size;

    g.pfn_spirv_to_dxil_free(&dxil_obj);
    return 0;
}

/* ============================================================================
 * Public API: Query buffer GPU address (for advanced use)
 * ============================================================================ */

uint64_t d3d12c_get_buffer_gpu_address(uint64_t handle) {
    HandleEntry *e = get_handle(handle);
    if (!e || e->type != HANDLE_BUFFER) return 0;
    return ID3D12Resource_GetGPUVirtualAddress(e->resource);
}
