//! Usage from the kernel's process list, which needs no admin rights, video memory from the
//! performance counters that Task Manager also shows, and per-process network from an event trace
//! that only works with admin rights.

use crate::{Details, Gpu, Proc, Sample, Total, Usage};
use std::{
    collections::HashMap,
    ffi::c_void,
    mem::size_of,
    os::windows::ffi::OsStrExt,
    slice,
    sync::{Arc, Mutex},
    thread,
    time::Instant,
};
use windows::{
    Wdk::System::SystemInformation::{NtQuerySystemInformation, SYSTEM_INFORMATION_CLASS, SystemProcessInformation, SystemProcessorPerformanceInformation},
    Win32::{
        Foundation::{CloseHandle, ERROR_ACCESS_DENIED, ERROR_SUCCESS, FALSE, HANDLE, HWND, LPARAM, POINT, RECT, STATUS_INFO_LENGTH_MISMATCH, TRUE, UNICODE_STRING},
        Graphics::{
            Dxgi::{CreateDXGIFactory1, IDXGIFactory1},
            Gdi::{GetMonitorInfoW, MONITOR_DEFAULTTONULL, MONITOR_DEFAULTTOPRIMARY, MONITORINFO, MonitorFromPoint},
        },
        NetworkManagement::{
            IpHelper::{FreeMibTable, GetIfTable2, IF_TYPE_SOFTWARE_LOOPBACK, MIB_IF_TABLE2},
            Ndis::IfOperStatusUp,
        },
        Security::{GetTokenInformation, TOKEN_ELEVATION, TOKEN_QUERY, TokenElevation},
        System::{
            Diagnostics::Etw::{
                CONTROLTRACE_HANDLE, CloseTrace, ControlTraceW, EVENT_CONTROL_CODE_ENABLE_PROVIDER, EVENT_RECORD, EVENT_TRACE_CONTROL_STOP, EVENT_TRACE_LOGFILEW, EVENT_TRACE_LOGFILEW_0,
                EVENT_TRACE_LOGFILEW_1, EVENT_TRACE_PROPERTIES, EVENT_TRACE_REAL_TIME_MODE, EnableTraceEx2, OpenTraceW, PROCESS_TRACE_MODE_EVENT_RECORD, PROCESS_TRACE_MODE_REAL_TIME, ProcessTrace,
                StartTraceW, WNODE_FLAG_TRACED_GUID,
            },
            Performance::{
                PDH_CSTATUS_NEW_DATA, PDH_CSTATUS_VALID_DATA, PDH_FMT_COUNTERVALUE, PDH_FMT_COUNTERVALUE_ITEM_W, PDH_FMT_LARGE, PDH_HCOUNTER, PDH_HQUERY, PDH_MORE_DATA,
                PdhAddEnglishCounterW, PdhCollectQueryData, PdhGetFormattedCounterArrayW, PdhGetFormattedCounterValue, PdhOpenQueryW,
            },
            Registry::{HKEY_LOCAL_MACHINE, REG_ROUTINE_FLAGS, RRF_RT_REG_MULTI_SZ, RRF_RT_REG_SZ, RegGetValueW},
            SystemInformation::{GetLogicalProcessorInformationEx, GlobalMemoryStatusEx, MEMORYSTATUSEX, RelationProcessorCore},
            Threading::{ALL_PROCESSOR_GROUPS, GetActiveProcessorCount, GetCurrentProcess, GetCurrentProcessId, OpenProcess, OpenProcessToken, PROCESS_TERMINATE, TerminateProcess},
            WindowsProgramming::SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION,
        },
        UI::{
            Shell::ShellExecuteW,
            WindowsAndMessaging::{EnumWindows, GetCursorPos, GetWindowRect, GetWindowThreadProcessId, IsWindowVisible, SW_SHOWNORMAL, SWP_NOSIZE, SWP_NOZORDER, SetWindowPos},
        },
    },
    core::{BOOL, GUID, PCWSTR, PWSTR, w},
};

/// The start of SYSTEM_PROCESS_INFORMATION, named as in System Informer's headers since most of it is undocumented.
#[repr(C)]
struct ProcessInfo {
    next_entry_offset: u32,
    number_of_threads: u32,
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
    inherited_from_unique_process_id: usize,
    _handle_count: u32,
    _session_id: u32,
    _unique_process_key: usize,
    _peak_virtual_size: usize,
    _virtual_size: usize,
    _page_fault_count: u32,
    _peak_working_set_size: usize,
    working_set_size: usize,
    _quota_peak_paged_pool_usage: usize,
    _quota_paged_pool_usage: usize,
    _quota_peak_non_paged_pool_usage: usize,
    _quota_non_paged_pool_usage: usize,
    pagefile_usage: usize,
    _peak_pagefile_usage: usize,
    _private_page_count: usize,
    _read_operation_count: u64,
    _write_operation_count: u64,
    _other_operation_count: u64,
    read_transfer_count: u64,
    write_transfer_count: u64,
    /// Kept so the struct's size matches the kernel's, placing the thread entries that follow it.
    _other_transfer_count: u64,
}

/// The start of SYSTEM_THREAD_INFORMATION, the per-thread entries the kernel writes right after each process.
#[repr(C)]
struct ThreadInfo {
    kernel_time: i64,
    user_time: i64,
    create_time: i64,
    _wait_time: u32,
    _start_address: usize,
    _unique_process: usize,
    unique_thread: usize,
    _priority: i32,
    _base_priority: i32,
    _context_switches: u32,
    _thread_state: u32,
    _wait_reason: u32,
}

pub struct Sampler {
    /// The process list, in u64 for its entries to be aligned.
    buf: Vec<u64>,
    /// The previous totals of each process, by id and creation time since ids get reused.
    cpu_times: HashMap<(u32, i64), u64>,
    /// The previous CPU total of each thread, to find the busiest one each sample.
    thread_times: HashMap<(u32, i64), u64>,
    /// The previous idle, kernel and user times of each CPU core, for its usage.
    core_times: Vec<(u64, u64, u64)>,
    disk_bytes: HashMap<(u32, i64), u64>,
    gpus: Option<Gpus>,
    system: Option<SystemMemory>,
    net: Option<NetTrace>,
    /// The previous network bytes of each process, from the trace.
    net_bytes: HashMap<u32, u64>,
    /// The previous bytes over all network interfaces.
    iface_bytes: Option<u64>,
    details: Details,
    prepared: bool,
    last: Instant,
}

impl Sampler {
    pub fn new() -> Self {
        Self {
            buf: vec![0; 1 << 17],
            cpu_times: HashMap::new(),
            thread_times: HashMap::new(),
            core_times: Vec::new(),
            disk_bytes: HashMap::new(),
            gpus: None,
            system: None,
            net: None,
            net_bytes: HashMap::new(),
            iface_bytes: None,
            details: Details::default(),
            prepared: false,
            last: Instant::now(),
        }
    }

    /// Sets up the GPU counters and the network trace, which together take about 200 ms, once.
    /// Called after the first sample so the process list, memory and CPU show without that wait.
    pub fn prepare(&mut self) {
        if !self.prepared {
            self.gpus = Gpus::new();
            self.system = SystemMemory::new();
            self.net = NetTrace::new();
            self.details.cpu = cpu_info();
            self.prepared = true;
        }
    }

    pub fn sample(&mut self) -> Sample {
        let secs = self.last.elapsed().as_secs_f64().max(0.001);
        self.last = Instant::now();
        let (gpus, mut vram) = self.gpus.as_mut().map(Gpus::sample).unwrap_or_default();
        let net_now = self.net.as_ref().map(NetTrace::totals);
        let mut procs = Vec::new();
        let mut cpu_deltas = Vec::new();
        let mut cpu_times = HashMap::new();
        let mut thread_times = HashMap::new();
        let mut disk_bytes = HashMap::new();
        // All CPU time goes to some process, idle time to the idle process with id 0,
        // so the deltas of all processes add up to the time that passed on all CPUs.
        let mut all_cpu_delta = 0;
        if self.query_processes() {
            let mut offset = 0;
            loop {
                // SAFETY: the kernel wrote the entries back to back, each one at the offset that the previous one gives.
                let base = unsafe { self.buf.as_ptr().cast::<u8>().add(offset) };
                let info = unsafe { &*base.cast::<ProcessInfo>() };
                let pid = info.unique_process_id as u32;
                let cpu_time = info.user_time + info.kernel_time;
                let cpu_delta = delta(&self.cpu_times, (pid, info.create_time), cpu_time);
                cpu_times.insert((pid, info.create_time), cpu_time);
                all_cpu_delta += cpu_delta;
                if pid != 0 {
                    let name = if info.image_name.Buffer.is_null() {
                        String::new()
                    } else {
                        String::from_utf16_lossy(unsafe { slice::from_raw_parts(info.image_name.Buffer.0, info.image_name.Length as usize / 2) })
                    };
                    let io = info.read_transfer_count + info.write_transfer_count;
                    let net = net_now.as_ref().map(|now| rate(delta(&self.net_bytes, pid, now.get(&pid).copied().unwrap_or(0)), secs)).unwrap_or(0);
                    // The busiest single thread, as a share of one core, reveals a core pegged by one thread even when the all-core share is low.
                    let threads = unsafe { base.add(size_of::<ProcessInfo>()).cast::<ThreadInfo>() };
                    let mut busiest = 0;
                    for t in 0..info.number_of_threads as usize {
                        // SAFETY: the kernel wrote number_of_threads entries right after the process.
                        let thread = unsafe { &*threads.add(t) };
                        let cpu = (thread.kernel_time + thread.user_time) as u64;
                        let tid = thread.unique_thread as u32;
                        busiest = busiest.max(delta(&self.thread_times, (tid, thread.create_time), cpu));
                        thread_times.insert((tid, thread.create_time), cpu);
                    }
                    let usage = Usage {
                        ram: info.working_set_private_size,
                        shared: (info.working_set_size as u64).saturating_sub(info.working_set_private_size),
                        commit: info.pagefile_usage as u64,
                        vram: vram.remove(&pid).unwrap_or_default(),
                        cpu: 0.0,
                        // Thread times are charged in clock ticks, sometimes late, so a busy thread's delta can exceed
                        // the interval by a few ticks; a thread can't use more than one core, so cap it there.
                        core: ((busiest as f64 * 100.0 / (secs * 1e7)) as f32).min(100.0),
                        disk: rate(delta(&self.disk_bytes, (pid, info.create_time), io), secs),
                        net,
                    };
                    disk_bytes.insert((pid, info.create_time), io);
                    procs.push(Proc { pid, parent: info.inherited_from_unique_process_id as u32, name, tip: "", usage });
                    cpu_deltas.push(cpu_delta);
                }
                if info.next_entry_offset == 0 {
                    break;
                }
                offset += info.next_entry_offset as usize;
            }
        }
        self.cpu_times = cpu_times;
        self.thread_times = thread_times;
        self.disk_bytes = disk_bytes;
        self.net_bytes = net_now.unwrap_or_default();
        for (proc, delta) in procs.iter_mut().zip(cpu_deltas) {
            proc.usage.cpu = delta as f32 * 100.0 / all_cpu_delta.max(1) as f32;
        }
        let (ram, commit) = memory();
        if let Some(system) = &mut self.system {
            let rows = system.rows(&ram, &commit, &procs, &gpus);
            procs.extend(rows);
        }
        let octets = interface_octets();
        let net = rate(self.iface_bytes.map(|prev| octets.saturating_sub(prev)).unwrap_or(0), secs);
        self.iface_bytes = Some(octets);
        let cores = self.cpu_cores();
        // Re-read every sample since a page file that Windows manages grows and shrinks.
        self.details.pagefile = pagefile_info();
        Sample { ram, commit, gpus, procs, net, net_per_process: self.net.is_some(), cores, details: self.details.clone() }
    }

    /// The busy share of each CPU core over the interval, where its kernel time counts idle time too.
    fn cpu_cores(&mut self) -> Vec<f32> {
        let mut buf = [SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION::default(); 64];
        let mut needed = 0;
        let size = (buf.len() * size_of::<SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION>()) as u32;
        let status = unsafe { NtQuerySystemInformation(SystemProcessorPerformanceInformation, buf.as_mut_ptr().cast(), size, &mut needed) };
        if status.is_err() {
            return Vec::new();
        }
        let count = needed as usize / size_of::<SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION>();
        let mut cores = Vec::with_capacity(count);
        let mut now = Vec::with_capacity(count);
        for (i, core) in buf[..count].iter().enumerate() {
            let (idle, kernel, user) = (core.IdleTime as u64, core.KernelTime as u64, core.UserTime as u64);
            let usage = match self.core_times.get(i) {
                Some(&(prev_idle, prev_kernel, prev_user)) => {
                    let total = kernel.saturating_sub(prev_kernel) + user.saturating_sub(prev_user);
                    let busy = total.saturating_sub(idle.saturating_sub(prev_idle));
                    if total > 0 { busy as f32 * 100.0 / total as f32 } else { 0.0 }
                }
                None => 0.0,
            };
            cores.push(usage);
            now.push((idle, kernel, user));
        }
        self.core_times = now;
        cores
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

/// How much a cumulative total grew since the previous sample, zero on the first sample so startup totals don't show as a spike.
fn delta<K: std::hash::Hash + Eq>(previous: &HashMap<K, u64>, key: K, now: u64) -> u64 {
    match previous.get(&key) {
        Some(&before) => now.saturating_sub(before),
        None if previous.is_empty() => 0,
        None => now,
    }
}

/// Bytes per second from an amount over an interval.
fn rate(bytes: u64, secs: f64) -> u64 {
    (bytes as f64 / secs) as u64
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
    fn sample(&mut self) -> (Vec<Gpu>, HashMap<u32, Vec<u64>>) {
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
        let gpus = self.adapters.iter().zip(used).map(|(adapter, used)| Gpu { name: adapter.name.clone(), vram: Total { used, total: adapter.memory } }).collect();
        (gpus, per_process)
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

/// The memory that no process owns, from the system's own counters, to list as rows of its own among the processes.
struct SystemMemory {
    query: PDH_HQUERY,
    nonpaged: PDH_HCOUNTER,
    paged: PDH_HCOUNTER,
    paged_resident: PDH_HCOUNTER,
    cache: PDH_HCOUNTER,
    drivers: PDH_HCOUNTER,
    code: PDH_HCOUNTER,
    modified: PDH_HCOUNTER,
    /// The system RAM that each GPU uses, missing on machines without GPU counters.
    gpu_shared: PDH_HCOUNTER,
    /// The pages Windows gave to the hypervisor, only on machines running one.
    hypervisor: Option<PDH_HCOUNTER>,
    buf: Vec<u64>,
}

impl SystemMemory {
    fn new() -> Option<Self> {
        let mut query = PDH_HQUERY::default();
        let mut counters = [PDH_HCOUNTER::default(); 7];
        let paths = [
            w!(r"\Memory\Pool Nonpaged Bytes"),
            w!(r"\Memory\Pool Paged Bytes"),
            w!(r"\Memory\Pool Paged Resident Bytes"),
            w!(r"\Memory\System Cache Resident Bytes"),
            w!(r"\Memory\System Driver Resident Bytes"),
            w!(r"\Memory\System Code Resident Bytes"),
            w!(r"\Memory\Modified Page List Bytes"),
        ];
        let (mut gpu_shared, mut hypervisor) = (PDH_HCOUNTER::default(), PDH_HCOUNTER::default());
        let hypervisor = unsafe {
            if PdhOpenQueryW(PCWSTR::null(), 0, &mut query) != 0 || paths.iter().zip(&mut counters).any(|(path, counter)| PdhAddEnglishCounterW(query, *path, 0, counter) != 0) {
                return None;
            }
            _ = PdhAddEnglishCounterW(query, w!(r"\GPU Adapter Memory(*)\Shared Usage"), 0, &mut gpu_shared);
            (PdhAddEnglishCounterW(query, w!(r"\Hyper-V Hypervisor Root Partition(*)\Deposited Pages"), 0, &mut hypervisor) == 0).then_some(hypervisor)
        };
        let [nonpaged, paged, paged_resident, cache, drivers, code, modified] = counters;
        Some(Self { query, nonpaged, paged, paged_resident, cache, drivers, code, modified, gpu_shared, hypervisor, buf: Vec::new() })
    }

    /// The rows, plus one for whatever neither they nor the processes account for, so the list adds up to the totals.
    fn rows(&mut self, ram: &Total, commit: &Total, procs: &[Proc], gpus: &[Gpu]) -> Vec<Proc> {
        unsafe { PdhCollectQueryData(self.query) };
        let (nonpaged, paged) = (value(self.nonpaged), value(self.paged));
        let gpu_shared = values(self.gpu_shared, &mut self.buf).iter().map(|(_, bytes)| bytes).sum();
        let mut rows = vec![
            row("Kernel non-paged pool", "Kernel and driver memory that always stays in RAM.", nonpaged, nonpaged),
            row("Kernel paged pool", "Kernel and driver memory that can move to the page file.", value(self.paged_resident), paged),
            row("GPU memory in RAM", "System RAM that the GPUs hold, like textures and buffers.", gpu_shared, 0),
            row("File cache", "Data of files in use, which Windows keeps in RAM.", value(self.cache), 0),
            row("Kernel and driver code", "The code of Windows' kernel and its drivers.", value(self.drivers) + value(self.code), 0),
            row("Modified pages", "Changed memory waiting to be written to disk.", value(self.modified), 0),
            row("Shared memory", "Memory that programs share with each other, counted once here rather than in any of them.", 0, shared_committed()),
        ];
        if let Some(hypervisor) = self.hypervisor {
            // Counted in pages, under the root partition, with a "_Total" instance repeating it.
            let pages: u64 = values(hypervisor, &mut self.buf).iter().filter(|(instance, _)| !instance.starts_with('_')).map(|(_, pages)| pages).sum();
            rows.push(row("Hypervisor", "Memory Windows gives to the hypervisor, which runs its virtualization-based security and virtual machines.", pages * 4096, 0));
        }
        let owned = |usage: fn(&Usage) -> u64| procs.iter().chain(&rows).map(|proc| usage(&proc.usage)).sum::<u64>();
        let mut other = row(
            "Other system memory",
            "Memory no program owns: pages shared between programs, memory that drivers lock, page tables and more.",
            ram.used.saturating_sub(owned(|usage| usage.ram)),
            commit.used.saturating_sub(owned(|usage| usage.commit)),
        );
        other.usage.vram = gpus.iter().enumerate().map(|(gpu, total)| total.vram.used.saturating_sub(procs.iter().filter_map(|proc| proc.usage.vram.get(gpu)).sum())).collect();
        rows.push(other);
        rows
    }
}

/// The start of SYSTEM_MEMORY_USAGE_INFORMATION.
#[repr(C)]
#[derive(Default)]
struct MemoryUsage {
    _total_physical: u64,
    _available: u64,
    _resident_available: i64,
    _committed: u64,
    shared_committed: u64,
    _commit_limit: u64,
    _peak_commitment: u64,
}

/// The commit charge of memory that programs share, like memory-mapped sections backed by the page file.
fn shared_committed() -> u64 {
    // SystemMemoryUsageInformation, which the bindings don't name.
    const MEMORY_USAGE: SYSTEM_INFORMATION_CLASS = SYSTEM_INFORMATION_CLASS(182);
    let mut usage = MemoryUsage::default();
    let mut needed = 0;
    let status = unsafe { NtQuerySystemInformation(MEMORY_USAGE, (&raw mut usage).cast(), size_of::<MemoryUsage>() as u32, &mut needed) };
    if status.is_ok() { usage.shared_committed } else { 0 }
}

/// A row of memory that no process owns.
fn row(name: &str, tip: &'static str, ram: u64, commit: u64) -> Proc {
    Proc { pid: 0, parent: 0, name: name.to_owned(), tip, usage: Usage { ram, commit, ..Default::default() } }
}

/// The value of a counter without instances, from the last collection.
fn value(counter: PDH_HCOUNTER) -> u64 {
    let mut value = PDH_FMT_COUNTERVALUE::default();
    if unsafe { PdhGetFormattedCounterValue(counter, PDH_FMT_LARGE, None, &mut value) } != 0 {
        return 0;
    }
    unsafe { value.Anonymous.largeValue as u64 }
}

/// RAM, and the commit charge against its limit.
fn memory() -> (Total, Total) {
    let mut status = MEMORYSTATUSEX { dwLength: size_of::<MEMORYSTATUSEX>() as u32, ..Default::default() };
    if unsafe { GlobalMemoryStatusEx(&mut status) }.is_err() {
        return Default::default();
    }
    let ram = Total { used: status.ullTotalPhys - status.ullAvailPhys, total: status.ullTotalPhys };
    let commit = Total { used: status.ullTotalPageFile - status.ullAvailPageFile, total: status.ullTotalPageFile };
    (ram, commit)
}

/// The CPU's name with its core and thread counts.
fn cpu_info() -> String {
    let name = registry(w!("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"), w!("ProcessorNameString"), RRF_RT_REG_SZ)
        .map(|name| String::from_utf16_lossy(&name).trim_matches(['\0', ' ']).to_owned())
        .unwrap_or_default();
    let threads = unsafe { GetActiveProcessorCount(ALL_PROCESSOR_GROUPS) };
    format!("{name}, {} cores / {threads} threads", physical_cores())
}

/// How many physical cores there are, as the kernel lists one processor-core entry per core.
fn physical_cores() -> usize {
    let mut len = 0;
    _ = unsafe { GetLogicalProcessorInformationEx(RelationProcessorCore, None, &mut len) };
    let mut buf = vec![0u64; (len as usize).div_ceil(8)];
    if unsafe { GetLogicalProcessorInformationEx(RelationProcessorCore, Some(buf.as_mut_ptr().cast()), &mut len) }.is_err() {
        return 0;
    }
    let bytes = unsafe { slice::from_raw_parts(buf.as_ptr().cast::<u8>(), len as usize) };
    let (mut cores, mut offset) = (0, 0);
    // Each entry starts with its relationship, then its own size in bytes.
    while let Some(size) = bytes.get(offset + 4..offset + 8).map(|size| u32::from_le_bytes(size.try_into().unwrap()) as usize).filter(|&size| size > 0) {
        cores += 1;
        offset += size;
    }
    cores
}

/// The start of SYSTEM_PAGEFILE_INFORMATION, one entry per page file.
#[repr(C)]
struct PageFileInfo {
    next_entry_offset: u32,
    /// In pages.
    total_size: u32,
}

/// The page file's current size, and whether it's fixed or grows.
fn pagefile_info() -> String {
    // SystemPageFileInformation, which the bindings don't name.
    const PAGE_FILES: SYSTEM_INFORMATION_CLASS = SYSTEM_INFORMATION_CLASS(18);
    // Windows uses 4 KB pages on every architecture it runs on.
    const PAGE: u64 = 4096;
    let mut buf = vec![0u64; 512];
    let mut needed = 0;
    let mut pages = 0;
    if unsafe { NtQuerySystemInformation(PAGE_FILES, buf.as_mut_ptr().cast(), (buf.len() * 8) as u32, &mut needed) }.is_ok() && needed > 0 {
        let mut offset = 0;
        loop {
            // SAFETY: the kernel wrote the entries back to back, each one at the offset that the previous one gives.
            let info = unsafe { &*buf.as_ptr().cast::<u8>().add(offset).cast::<PageFileInfo>() };
            pages += u64::from(info.total_size);
            if info.next_entry_offset == 0 {
                break;
            }
            offset += info.next_entry_offset as usize;
        }
    }
    if pages == 0 {
        return "No page file".to_owned();
    }
    let size = crate::size(pages * PAGE);
    // Entries like "C:\pagefile.sys 4096 8192", with the initial and maximum sizes in MB.
    let Some(file) = registry(w!("SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Memory Management"), w!("PagingFiles"), RRF_RT_REG_MULTI_SZ)
        .and_then(|files| String::from_utf16_lossy(&files).split('\0').find(|file| !file.is_empty()).map(str::to_owned))
    else {
        return format!("Page file {size}");
    };
    let mut limits = file.split_whitespace().skip(1).map(|mb| mb.parse::<u64>().unwrap_or(0));
    let (initial, max) = (limits.next().unwrap_or(0), limits.next().unwrap_or(0));
    // Without sizes, or with zero ones, Windows manages it.
    let mode = match (initial, max) {
        (_, 0) => "auto-growing".to_owned(),
        (initial, max) if initial == max => "fixed".to_owned(),
        (_, max) => format!("grows to {}", crate::size(max << 20)),
    };
    format!("Page file {size}, {mode}")
}

/// A string or multi-string value under HKEY_LOCAL_MACHINE, as UTF-16 including its terminating nulls.
fn registry(key: PCWSTR, value: PCWSTR, kind: REG_ROUTINE_FLAGS) -> Option<Vec<u16>> {
    let mut size = 0u32;
    if unsafe { RegGetValueW(HKEY_LOCAL_MACHINE, key, value, kind, None, None, Some(&raw mut size)) } != ERROR_SUCCESS {
        return None;
    }
    let mut buf = vec![0u16; size as usize / 2];
    if unsafe { RegGetValueW(HKEY_LOCAL_MACHINE, key, value, kind, None, Some(buf.as_mut_ptr().cast()), Some(&raw mut size)) } != ERROR_SUCCESS {
        return None;
    }
    buf.truncate(size as usize / 2);
    Some(buf)
}

/// The bytes sent and received over all real network interfaces, so far.
fn interface_octets() -> u64 {
    let mut table: *mut MIB_IF_TABLE2 = std::ptr::null_mut();
    if unsafe { GetIfTable2(&mut table) } != ERROR_SUCCESS || table.is_null() {
        return 0;
    }
    let rows = unsafe { slice::from_raw_parts((*table).Table.as_ptr(), (*table).NumEntries as usize) };
    let sum = rows
        .iter()
        .filter(|row| row.Type != IF_TYPE_SOFTWARE_LOOPBACK && row.OperStatus == IfOperStatusUp)
        .map(|row| row.InOctets + row.OutOctets)
        .sum();
    unsafe { FreeMibTable(table as _) };
    sum
}

/// Microsoft-Windows-Kernel-Network, the provider of the network events, by its GUID.
const KERNEL_NETWORK: GUID = GUID::from_values(0x7dd42a49, 0x5329, 0x4832, [0x8d, 0xfd, 0x43, 0xd9, 0x79, 0x15, 0x3a, 0x88]);
/// Both IPv4 and IPv6 datagrams.
const IP_KEYWORDS: u64 = 0x10 | 0x20;

/// A real-time trace of the per-process network bytes, which the kernel only reports to an admin.
struct NetTrace {
    session: CONTROLTRACE_HANDLE,
    /// Reused to stop the session.
    props: Vec<u8>,
    /// Kept alive while the trace refers to it.
    _name: Vec<u16>,
    /// The bytes sent and received by each process so far, shared with the consuming thread.
    counts: Arc<Mutex<HashMap<u32, (u64, u64)>>>,
    /// A leaked reference to `counts` passed to the trace callback, reclaimed on stop.
    context: *const Mutex<HashMap<u32, (u64, u64)>>,
    consumer: Option<thread::JoinHandle<()>>,
}

impl NetTrace {
    fn new() -> Option<Self> {
        let counts = Arc::new(Mutex::new(HashMap::new()));
        let context = Arc::into_raw(counts.clone());
        let mut name: Vec<u16> = "tuskManager".encode_utf16().chain([0]).collect();
        let (session, mut props) = start_session(&name)?;
        // Verbose level, above the informational level of the network events.
        unsafe { _ = EnableTraceEx2(session, &KERNEL_NETWORK, EVENT_CONTROL_CODE_ENABLE_PROVIDER.0, 5, IP_KEYWORDS, 0, 0, None) };

        let mut logfile = EVENT_TRACE_LOGFILEW {
            LoggerName: PWSTR(name.as_mut_ptr()),
            Anonymous1: EVENT_TRACE_LOGFILEW_0 { ProcessTraceMode: PROCESS_TRACE_MODE_REAL_TIME | PROCESS_TRACE_MODE_EVENT_RECORD },
            Anonymous2: EVENT_TRACE_LOGFILEW_1 { EventRecordCallback: Some(on_event) },
            Context: context as *mut c_void,
            ..Default::default()
        };
        let trace = unsafe { OpenTraceW(&mut logfile) };
        if trace.Value == u64::MAX {
            unsafe { _ = ControlTraceW(session, PCWSTR::null(), props.as_mut_ptr() as *mut _, EVENT_TRACE_CONTROL_STOP) };
            unsafe { drop(Arc::from_raw(context)) };
            return None;
        }
        // ProcessTrace blocks until the session is stopped, feeding events to the callback.
        let consumer = thread::spawn(move || unsafe {
            _ = ProcessTrace(&[trace], None, None);
            _ = CloseTrace(trace);
        });
        Some(Self { session, props, _name: name, counts, context, consumer: Some(consumer) })
    }

    /// The bytes sent plus received by each process so far.
    fn totals(&self) -> HashMap<u32, u64> {
        self.counts.lock().map(|counts| counts.iter().map(|(&pid, &(sent, received))| (pid, sent + received)).collect()).unwrap_or_default()
    }
}

impl Drop for NetTrace {
    fn drop(&mut self) {
        unsafe { _ = ControlTraceW(self.session, PCWSTR::null(), self.props.as_mut_ptr() as *mut _, EVENT_TRACE_CONTROL_STOP) };
        if let Some(consumer) = self.consumer.take() {
            let _ = consumer.join();
        }
        unsafe { drop(Arc::from_raw(self.context)) };
    }
}

/// Starts a real-time session by the given name, stopping a stale one left by a previous run. `None` without admin rights.
fn start_session(name: &[u16]) -> Option<(CONTROLTRACE_HANDLE, Vec<u8>)> {
    for retry in [false, true] {
        let mut props = vec![0u8; size_of::<EVENT_TRACE_PROPERTIES>() + name.len() * 2];
        let header = props.as_mut_ptr() as *mut EVENT_TRACE_PROPERTIES;
        unsafe {
            (*header).Wnode.BufferSize = props.len() as u32;
            (*header).Wnode.Flags = WNODE_FLAG_TRACED_GUID;
            (*header).LogFileMode = EVENT_TRACE_REAL_TIME_MODE;
            (*header).LoggerNameOffset = size_of::<EVENT_TRACE_PROPERTIES>() as u32;
        }
        let mut session = CONTROLTRACE_HANDLE::default();
        let status = unsafe { StartTraceW(&mut session, PCWSTR(name.as_ptr()), header) };
        if status == ERROR_SUCCESS {
            return Some((session, props));
        }
        if status == ERROR_ACCESS_DENIED || retry {
            return None;
        }
        unsafe { _ = ControlTraceW(CONTROLTRACE_HANDLE::default(), PCWSTR(name.as_ptr()), header, EVENT_TRACE_CONTROL_STOP) };
    }
    None
}

/// Adds a network event's bytes to the process that sent or received them.
unsafe extern "system" fn on_event(record: *mut EVENT_RECORD) {
    let record = unsafe { &*record };
    let sent = matches!(record.EventHeader.EventDescriptor.Id, 10 | 26 | 42 | 58);
    let received = matches!(record.EventHeader.EventDescriptor.Id, 11 | 27 | 43 | 59);
    if (!sent && !received) || record.UserDataLength < 8 || record.UserData.is_null() {
        return;
    }
    // The event starts with the process id and the byte count.
    let data = record.UserData as *const u8;
    let pid = unsafe { data.cast::<u32>().read_unaligned() };
    let bytes = unsafe { data.add(4).cast::<u32>().read_unaligned() } as u64;
    let counts = unsafe { &*(record.UserContext as *const Mutex<HashMap<u32, (u64, u64)>>) };
    if let Ok(mut counts) = counts.lock() {
        let entry = counts.entry(pid).or_insert((0, 0));
        if sent {
            entry.0 += bytes;
        } else {
            entry.1 += bytes;
        }
    }
}

/// Terminates each of the given processes, those it has the rights to open.
pub fn end(pids: &[u32]) {
    for &pid in pids {
        unsafe {
            if let Ok(handle) = OpenProcess(PROCESS_TERMINATE, false, pid) {
                let _ = TerminateProcess(handle, 1);
                let _ = CloseHandle(handle);
            }
        }
    }
}

/// Whether to offer relaunching as admin, which is what per-process network needs.
pub fn offer_admin() -> bool {
    !elevated()
}

fn elevated() -> bool {
    unsafe {
        let mut token = HANDLE::default();
        if OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token).is_err() {
            return false;
        }
        let mut elevation = TOKEN_ELEVATION::default();
        let mut size = 0;
        let ok = GetTokenInformation(token, TokenElevation, Some(&mut elevation as *mut _ as *mut c_void), size_of::<TOKEN_ELEVATION>() as u32, &mut size).is_ok();
        let _ = CloseHandle(token);
        ok && elevation.TokenIsElevated != 0
    }
}

/// Positions the window on startup, since eframe misplaces saved positions across mixed-DPI monitors: back to its saved
/// spot when that's on the primary monitor or the one the user is working on, otherwise centered on the primary. This keeps
/// it off a vanished or VR virtual monitor. `saved` is the saved top-left in pixels. Returns true once done.
pub fn place_window(saved: Option<(i32, i32)>) -> bool {
    let Some(hwnd) = main_window() else { return false };
    unsafe {
        let mut rect = RECT::default();
        if GetWindowRect(hwnd, &mut rect).is_err() {
            return false;
        }
        let (width, height) = (rect.right - rect.left, rect.bottom - rect.top);
        // eframe sizes the window a frame or two after creation; wait for its real size before placing it.
        if width < 200 || height < 150 {
            return false;
        }
        let primary = MonitorFromPoint(POINT::default(), MONITOR_DEFAULTTOPRIMARY);
        let mut cursor = POINT::default();
        let _ = GetCursorPos(&mut cursor);
        let under_cursor = MonitorFromPoint(cursor, MONITOR_DEFAULTTONULL);
        // Restore the saved spot only if it sits on the primary monitor or the one the user is working on.
        if let Some((x, y)) = saved {
            let saved_monitor = MonitorFromPoint(POINT { x, y }, MONITOR_DEFAULTTONULL);
            if !saved_monitor.is_invalid() && (saved_monitor.0 == primary.0 || saved_monitor.0 == under_cursor.0) {
                let _ = SetWindowPos(hwnd, None, x, y, 0, 0, SWP_NOSIZE | SWP_NOZORDER);
                return true;
            }
        }
        let mut info = MONITORINFO { cbSize: size_of::<MONITORINFO>() as u32, ..Default::default() };
        if GetMonitorInfoW(primary, &mut info).as_bool() {
            let work = info.rcWork;
            let x = work.left + ((work.right - work.left - width) / 2).max(0);
            let y = work.top + ((work.bottom - work.top - height) / 2).max(0);
            let _ = SetWindowPos(hwnd, None, x, y, 0, 0, SWP_NOSIZE | SWP_NOZORDER);
        }
    }
    true
}

/// This process's visible top-level window.
fn main_window() -> Option<HWND> {
    unsafe extern "system" fn visit(hwnd: HWND, out: LPARAM) -> BOOL {
        let mut pid = 0;
        unsafe { GetWindowThreadProcessId(hwnd, Some(&mut pid)) };
        if pid == unsafe { GetCurrentProcessId() } && unsafe { IsWindowVisible(hwnd) }.as_bool() {
            unsafe { *(out.0 as *mut HWND) = hwnd };
            return FALSE;
        }
        TRUE
    }
    let mut hwnd = HWND::default();
    let _ = unsafe { EnumWindows(Some(visit), LPARAM(&mut hwnd as *mut _ as isize)) };
    (!hwnd.is_invalid()).then_some(hwnd)
}

/// Starts a new elevated instance, prompting for admin rights. The caller should then exit.
pub fn relaunch_as_admin() {
    let Ok(exe) = std::env::current_exe() else { return };
    let exe: Vec<u16> = exe.as_os_str().encode_wide().chain([0]).collect();
    unsafe { ShellExecuteW(None, w!("runas"), PCWSTR(exe.as_ptr()), PCWSTR::null(), PCWSTR::null(), SW_SHOWNORMAL) };
}
