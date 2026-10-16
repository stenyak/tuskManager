//! Usage from the kernel's process list, which needs no admin rights,
//! and video memory from the performance counters that Task Manager also shows.

use crate::{Proc, Sample, Total, Usage};
use std::{collections::HashMap, mem::size_of, slice};
use windows::{
    Wdk::System::SystemInformation::{NtQuerySystemInformation, SystemProcessInformation},
    Win32::{
        Foundation::{STATUS_INFO_LENGTH_MISMATCH, UNICODE_STRING},
        Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1},
        System::{
            Performance::{
                PDH_CSTATUS_NEW_DATA, PDH_CSTATUS_VALID_DATA, PDH_FMT_COUNTERVALUE_ITEM_W, PDH_FMT_LARGE, PDH_HCOUNTER, PDH_HQUERY, PDH_MORE_DATA, PdhAddEnglishCounterW,
                PdhCollectQueryData, PdhGetFormattedCounterArrayW, PdhOpenQueryW,
            },
            SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX},
        },
    },
    core::{PCWSTR, w},
};

/// The start of SYSTEM_PROCESS_INFORMATION, named as in System Informer's headers since most of it is undocumented.
#[repr(C)]
struct ProcessInfo {
    next_entry_offset: u32,
    _number_of_threads: u32,
    working_set_private_size: u64,
    _hard_fault_count: u32,
    _number_of_threads_high_watermark: u32,
    _cycle_time: u64,
    create_time: i64,
    user_time: u64,
    kernel_time: u64,
    image_name: UNICODE_STRING,
    _base_priority: i32,
    unique_process_id: usize,
    _inherited_from_unique_process_id: usize,
    _handle_count: u32,
    _session_id: u32,
    _unique_process_key: usize,
    _peak_virtual_size: usize,
    _virtual_size: usize,
    _page_fault_count: u32,
    _peak_working_set_size: usize,
    _working_set_size: usize,
    _quota_peak_paged_pool_usage: usize,
    _quota_paged_pool_usage: usize,
    _quota_peak_non_paged_pool_usage: usize,
    _quota_non_paged_pool_usage: usize,
    pagefile_usage: usize,
}

pub struct Sampler {
    /// The process list, in u64 for its entries to be aligned.
    buf: Vec<u64>,
    /// The CPU time of each process at the previous sample, by id and creation time since ids get reused.
    cpu_times: HashMap<(u32, i64), u64>,
    gpus: Option<Gpus>,
}

impl Sampler {
    pub fn new() -> Self {
        Self { buf: vec![0; 1 << 17], cpu_times: HashMap::new(), gpus: Gpus::new() }
    }

    pub fn sample(&mut self) -> Sample {
        let (gpus, mut vram) = self.gpus.as_mut().map(Gpus::sample).unwrap_or_default();
        let mut procs = Vec::new();
        let mut cpu_deltas = Vec::new();
        let mut cpu_times = HashMap::new();
        // All CPU time goes to some process, idle time to the idle process with id 0,
        // so the deltas of all processes add up to the time that passed on all CPUs.
        let mut all_cpu_delta = 0;
        if self.query_processes() {
            let mut offset = 0;
            loop {
                // SAFETY: the kernel wrote the entries back to back, each one at the offset that the previous one gives.
                let info = unsafe { &*self.buf.as_ptr().cast::<u8>().add(offset).cast::<ProcessInfo>() };
                let pid = info.unique_process_id as u32;
                let cpu_time = info.user_time + info.kernel_time;
                let delta = match self.cpu_times.get(&(pid, info.create_time)) {
                    Some(&previous) => cpu_time.saturating_sub(previous),
                    None if self.cpu_times.is_empty() => 0,
                    None => cpu_time,
                };
                cpu_times.insert((pid, info.create_time), cpu_time);
                all_cpu_delta += delta;
                if pid != 0 {
                    let name = if info.image_name.Buffer.is_null() {
                        String::new()
                    } else {
                        String::from_utf16_lossy(unsafe { slice::from_raw_parts(info.image_name.Buffer.0, info.image_name.Length as usize / 2) })
                    };
                    let usage = Usage { ram: info.working_set_private_size, commit: info.pagefile_usage as u64, vram: vram.remove(&pid).unwrap_or_default(), cpu: 0.0 };
                    procs.push(Proc { pid, name, usage });
                    cpu_deltas.push(delta);
                }
                if info.next_entry_offset == 0 {
                    break;
                }
                offset += info.next_entry_offset as usize;
            }
        }
        self.cpu_times = cpu_times;
        for (proc, delta) in procs.iter_mut().zip(cpu_deltas) {
            proc.usage.cpu = delta as f32 * 100.0 / all_cpu_delta.max(1) as f32;
        }
        Sample { memory: memory(), gpus, procs }
    }

    /// Fills the buffer with the kernel's list of processes, growing it as needed.
    fn query_processes(&mut self) -> bool {
        loop {
            let mut needed = 0;
            let size = (self.buf.len() * 8) as u32;
            let status = unsafe { NtQuerySystemInformation(SystemProcessInformation, self.buf.as_mut_ptr().cast(), size, &mut needed) };
            if status != STATUS_INFO_LENGTH_MISMATCH {
                return status.is_ok();
            }
            // With room for processes that start before the next try.
            self.buf.resize(needed as usize / 8 + (1 << 14), 0);
        }
    }
}

/// The dedicated video memory that each GPU and process uses.
struct Gpus {
    query: PDH_HQUERY,
    per_process: PDH_HCOUNTER,
    per_gpu: PDH_HCOUNTER,
    adapters: Vec<Adapter>,
    buf: Vec<u64>,
}

struct Adapter {
    name: String,
    memory: u64,
    /// How the counter instance names of this GPU start, like "luid_0x00000000_0x0000d1a5".
    luid: String,
}

impl Gpus {
    fn new() -> Option<Self> {
        let mut query = PDH_HQUERY::default();
        let (mut per_process, mut per_gpu) = (PDH_HCOUNTER::default(), PDH_HCOUNTER::default());
        // The English names work on any Windows language.
        let ok = unsafe {
            PdhOpenQueryW(PCWSTR::null(), 0, &mut query) == 0
                && PdhAddEnglishCounterW(query, w!(r"\GPU Process Memory(*)\Dedicated Usage"), 0, &mut per_process) == 0
                && PdhAddEnglishCounterW(query, w!(r"\GPU Adapter Memory(*)\Dedicated Usage"), 0, &mut per_gpu) == 0
                && PdhCollectQueryData(query) == 0
        };
        if !ok {
            return None;
        }
        let mut gpus = Self { query, per_process, per_gpu, adapters: adapters(), buf: Vec::new() };
        // Indirect displays, like the virtual monitors of VR headsets, show up as one more adapter
        // named like the GPU that renders them, but without memory counters of their own.
        let instances = values(per_gpu, &mut gpus.buf);
        gpus.adapters.retain(|adapter| instances.iter().any(|(instance, _)| instance.to_ascii_lowercase().starts_with(&adapter.luid)));
        (!gpus.adapters.is_empty()).then_some(gpus)
    }

    /// Each GPU's use, and each process's use of each GPU by process id.
    fn sample(&mut self) -> (Vec<Total>, HashMap<u32, Vec<u64>>) {
        unsafe { PdhCollectQueryData(self.query) };
        let mut per_process: HashMap<u32, Vec<u64>> = HashMap::new();
        for (instance, bytes) in values(self.per_process, &mut self.buf) {
            // Like "pid_1234_luid_0x00000000_0x0000d1a5_phys_0".
            let Some((pid, luid)) = instance.strip_prefix("pid_").and_then(|rest| rest.split_once('_')) else { continue };
            if let (Ok(pid), Some(gpu)) = (pid.parse(), self.gpu(luid)) {
                per_process.entry(pid).or_insert_with(|| vec![0; self.adapters.len()])[gpu] += bytes;
            }
        }
        let mut used = vec![0; self.adapters.len()];
        for (instance, bytes) in values(self.per_gpu, &mut self.buf) {
            if let Some(gpu) = self.gpu(&instance) {
                used[gpu] += bytes;
            }
        }
        let totals = self.adapters.iter().zip(used).map(|(adapter, used)| Total { name: adapter.name.clone(), used, total: adapter.memory }).collect();
        (totals, per_process)
    }

    fn gpu(&self, instance: &str) -> Option<usize> {
        let instance = instance.to_ascii_lowercase();
        self.adapters.iter().position(|adapter| instance.starts_with(&adapter.luid))
    }
}

/// The GPUs with memory of their own, which leaves out software renderers, biggest first.
fn adapters() -> Vec<Adapter> {
    let Ok(factory) = (unsafe { CreateDXGIFactory1::<IDXGIFactory1>() }) else { return Vec::new() };
    let mut adapters: Vec<_> = (0..)
        .map_while(|i| unsafe { factory.EnumAdapters1(i) }.ok())
        .filter_map(|adapter| unsafe { adapter.GetDesc1() }.ok())
        .filter(|desc| desc.DedicatedVideoMemory > 0)
        .map(|desc| {
            let len = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
            Adapter {
                name: String::from_utf16_lossy(&desc.Description[..len]),
                memory: desc.DedicatedVideoMemory as u64,
                luid: format!("luid_0x{:08x}_0x{:08x}", desc.AdapterLuid.HighPart, desc.AdapterLuid.LowPart),
            }
        })
        .collect();
    adapters.sort_by_key(|adapter| std::cmp::Reverse(adapter.memory));
    adapters
}

/// The value of each instance of a counter with a wildcard instance, from the last collection.
fn values(counter: PDH_HCOUNTER, buf: &mut Vec<u64>) -> Vec<(String, u64)> {
    let (mut size, mut count) = (0, 0);
    unsafe {
        if PdhGetFormattedCounterArrayW(counter, PDH_FMT_LARGE, &mut size, &mut count, None) != PDH_MORE_DATA {
            return Vec::new();
        }
        buf.resize((size as usize).div_ceil(8), 0);
        if PdhGetFormattedCounterArrayW(counter, PDH_FMT_LARGE, &mut size, &mut count, Some(buf.as_mut_ptr().cast())) != 0 {
            return Vec::new();
        }
        slice::from_raw_parts(buf.as_ptr().cast::<PDH_FMT_COUNTERVALUE_ITEM_W>(), count as usize)
            .iter()
            .filter(|item| matches!(item.FmtValue.CStatus, PDH_CSTATUS_VALID_DATA | PDH_CSTATUS_NEW_DATA))
            .map(|item| (item.szName.to_string().unwrap_or_default(), item.FmtValue.Anonymous.largeValue as u64))
            .collect()
    }
}

fn memory() -> Vec<Total> {
    let mut status = MEMORYSTATUSEX { dwLength: size_of::<MEMORYSTATUSEX>() as u32, ..Default::default() };
    if unsafe { GlobalMemoryStatusEx(&mut status) }.is_err() {
        return Vec::new();
    }
    vec![
        Total { name: "RAM".to_owned(), used: status.ullTotalPhys - status.ullAvailPhys, total: status.ullTotalPhys },
        Total { name: "Commit".to_owned(), used: status.ullTotalPageFile - status.ullAvailPageFile, total: status.ullTotalPageFile },
    ]
}
