# Minimal Vulkan compute test via wgpu
import numpy as np
import sys
sys.path.insert(0, '/home/mukshud/claude_wsl/new/claude_rotifer/operonfold')

# Force Dozen backend
import os
os.environ['MESA_VK_VERSION_OVERRIDE'] = '1.0'
os.environ['VK_ICD_FILENAMES'] = '/home/mukshud/mesa-dozen-install/share/vulkan/icd.d/dzn_icd.aarch64.json'

try:
    import wgpu
    device = wgpu.utils.get_default_device()
    print(f"wgpu adapter: {device.adapter.info}")
    
    # Simple compute: add two buffers
    shader_code = """
@group(0) @binding(0) var<storage,read_write> a: array<f32>;
@group(0) @binding(1) var<storage,read_write> b: array<f32>;
@group(0) @binding(2) var<storage,read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&out)) {
        out[idx] = a[idx] + b[idx];
    }
}
"""
    shader = device.create_shader_module(code=shader_code)
    
    N = 256
    a = np.ones(N, dtype=np.float32) * 3.0
    b = np.ones(N, dtype=np.float32) * 7.0
    
    buf_a = device.create_buffer_with_data(data=a.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    buf_b = device.create_buffer_with_data(data=b.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    buf_out = device.create_buffer(size=N*4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    
    binding_layout = [
        {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
        {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
        {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
    ]
    bg_layout = device.create_bind_group_layout(entries=binding_layout)
    
    bindings = [
        {"binding": 0, "resource": {"buffer": buf_a}},
        {"binding": 1, "resource": {"buffer": buf_b}},
        {"binding": 2, "resource": {"buffer": buf_out}},
    ]
    bg = device.create_bind_group(layout=bg_layout, entries=bindings)
    
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bg_layout])
    pipeline = device.create_compute_pipeline(layout=pipeline_layout, compute={"module": shader, "entry_point": "main"})
    
    encoder = device.create_command_encoder()
    cpass = encoder.begin_compute_pass()
    cpass.set_pipeline(pipeline)
    cpass.set_bind_group(0, bg)
    cpass.dispatch_workgroups(N // 64)
    cpass.end()
    device.queue.submit([encoder.finish()])
    
    result = np.frombuffer(device.queue.read_buffer(buf_out), dtype=np.float32)
    expected = 10.0
    print(f"Result[0]={result[0]}, expected={expected}, all_correct={np.allclose(result, expected)}")
    
except Exception as e:
    print(f"Error: {e}")
