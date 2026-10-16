//! Usage from the kernel's process list, which needs no admin rights, video memory from the
//! performance counters that Task Manager also shows, and per-process network from an event trace
//! that only works with admin rights.

use crate::{Gpu, Proc, Sample, Total, Usage};
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
    Wdk::System::SystemInformation::{NtQuerySystemInformation, SystemProcessInformation},
    Win32::{
        Foundation::{CloseHandle, ERROR_ACCESS_DENIED, ERROR_SUCCESS, HANDLE, STATUS_INFO_LENGTH_MISMATCH, UNICODE_STRING},
        Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1},
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
                PDH_CSTATUS_NEW_DATA, PDH_CSTATUS_VALID_DATA, PDH_FMT_COUNTERVALUE_ITEM_W, PDH_FMT_LARGE, PDH_HCOUNTER, PDH_HQUERY, PDH_MORE_DATA, PdhAddEnglishCounterW,
                PdhCollectQueryData, PdhGetFormattedCounterArrayW, PdhOpenQueryW,
            },
            SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX},
            Threading::{GetCurrentProcess, OpenProcessToken},
        },
        UI::{Shell::ShellExecuteW, WindowsAndMessaging::SW_SHOWNORMAL},
    },
    core::{GUID, PCWSTR, PWSTR, w},
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
    _peak_pagefile_usage: usize,
    _private_page_count: usize,
    _read_operation_count: u64,
    _write_operation_count: u64,
    _other_operation_count: u64,
    read_transfer_count: u64,
    write_transfer_count: u64,
}

pub struct Sampler {
    /// The process list, in u64 for its entries to be aligned.
    buf: Vec<u64>,
    /// The previous totals of each process, by id and creation time since ids get reused.
    cpu_times: HashMap<(u32, i64), u64>,
    disk_bytes: HashMap<(u32, i64), u64>,
    gpus: Option<Gpus>,
    net: Option<NetTrace>,
    /// The previous network bytes of each process, from the trace.
    net_bytes: HashMap<u32, u64>,
    /// The previous bytes over all network interfaces.
    iface_bytes: Option<u64>,
    last: Instant,
}

impl Sampler {
    pub fn new() -> Self {
        Self {
            buf: vec![0; 1 << 17],
            cpu_times: HashMap::new(),
            disk_bytes: HashMap::new(),
            gpus: Gpus::new(),
            net: NetTrace::new(),
            net_bytes: HashMap::new(),
            iface_bytes: None,
            last: Instant::now(),
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
        let mut disk_bytes = HashMap::new();
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
                    let usage = Usage {
                        ram: info.working_set_private_size,
                        commit: info.pagefile_usage as u64,
                        vram: vram.remove(&pid).unwrap_or_default(),
                        cpu: 0.0,
                        disk: rate(delta(&self.disk_bytes, (pid, info.create_time), io), secs),
                        net,
                    };
                    disk_bytes.insert((pid, info.create_time), io);
                    procs.push(Proc { pid, name, usage });
                    cpu_deltas.push(cpu_delta);
                }
                if info.next_entry_offset == 0 {
                    break;
                }
                offset += info.next_entry_offset as usize;
            }
        }
        self.cpu_times = cpu_times;
        self.disk_bytes = disk_bytes;
        self.net_bytes = net_now.unwrap_or_default();
        for (proc, delta) in procs.iter_mut().zip(cpu_deltas) {
            proc.usage.cpu = delta as f32 * 100.0 / all_cpu_delta.max(1) as f32;
        }
        let (ram, commit) = memory();
        let octets = interface_octets();
        let net = rate(self.iface_bytes.map(|prev| octets.saturating_sub(prev)).unwrap_or(0), secs);
        self.iface_bytes = Some(octets);
        Sample { ram, commit, gpus, procs, net, net_per_process: self.net.is_some() }
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

/// Starts a new elevated instance, prompting for admin rights. The caller should then exit.
pub fn relaunch_as_admin() {
    let Ok(exe) = std::env::current_exe() else { return };
    let exe: Vec<u16> = exe.as_os_str().encode_wide().chain([0]).collect();
    unsafe { ShellExecuteW(None, w!("runas"), PCWSTR(exe.as_ptr()), PCWSTR::null(), PCWSTR::null(), SW_SHOWNORMAL) };
}
