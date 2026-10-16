//! Usage from sysinfo, plus the private memory and video memory of each process where Linux exposes them.

use crate::{Gpu, Proc, Sample, Total, Usage};
use std::collections::HashMap;
use sysinfo::{CpuRefreshKind, ProcessRefreshKind, ProcessesToUpdate, System, UpdateKind};
#[cfg(target_os = "linux")]
use {
    nvml_wrapper::{Nvml, enums::device::UsedGpuMemory},
    std::{collections::HashSet, fs},
};

pub struct Sampler {
    system: System,
    #[cfg(target_os = "linux")]
    nvml: Option<Nvml>,
}

impl Sampler {
    pub fn new() -> Self {
        let mut system = System::new();
        system.refresh_cpu_list(CpuRefreshKind::nothing());
        Self {
            system,
            #[cfg(target_os = "linux")]
            nvml: Nvml::init().ok(),
        }
    }

    pub fn sample(&mut self) -> Sample {
        let refresh = ProcessRefreshKind::nothing().with_cpu().with_memory().with_exe(UpdateKind::OnlyIfNotSet).without_tasks();
        self.system.refresh_processes_specifics(ProcessesToUpdate::All, true, refresh);
        self.system.refresh_memory();
        let (gpus, mut vram) = self.gpus();
        // sysinfo gives the CPU use in percent of one core.
        let cpus = self.system.cpus().len().max(1) as f32;
        let procs = self
            .system
            .processes()
            .values()
            // Leaves out kernel threads.
            .filter(|process| process.thread_kind().is_none())
            .map(|process| {
                let pid = process.pid().as_u32();
                let name = process.exe().and_then(|exe| exe.file_name()).unwrap_or(process.name());
                let (ram, commit) = private_memory(pid).unwrap_or((process.memory(), 0));
                let usage = Usage { ram, commit, vram: vram.remove(&pid).unwrap_or_default(), cpu: process.cpu_usage() / cpus };
                Proc { pid, name: name.to_string_lossy().into_owned(), usage }
            })
            .collect();
        let ram = Total { used: self.system.used_memory(), total: self.system.total_memory() };
        // There's no commit limit, swap is what extends RAM.
        let commit = Total { used: ram.used + self.system.used_swap(), total: ram.total + self.system.total_swap() };
        Sample { ram, commit, gpus, procs }
    }

    /// Each GPU's use, and each process's use of each GPU by process id.
    #[cfg(target_os = "linux")]
    fn gpus(&self) -> (Vec<Gpu>, HashMap<u32, Vec<u64>>) {
        let mut gpus = Vec::new();
        let mut vram: HashMap<u32, Vec<u64>> = HashMap::new();
        let mut set = |pid: u32, gpu: usize, bytes: u64| {
            let usage = vram.entry(pid).or_default();
            usage.resize(usage.len().max(gpu + 1), 0);
            usage[gpu] = usage[gpu].max(bytes);
        };
        // NVIDIA's driver only tells through NVML.
        if let Some(nvml) = &self.nvml {
            for device in (0..nvml.device_count().unwrap_or(0)).filter_map(|i| nvml.device_by_index(i).ok()) {
                let (Ok(name), Ok(memory)) = (device.name(), device.memory_info()) else { continue };
                // Processes that do both graphics and compute are in both lists, with the same memory.
                let graphics = device.running_graphics_processes().unwrap_or_default();
                for process in graphics.into_iter().chain(device.running_compute_processes().unwrap_or_default()) {
                    if let UsedGpuMemory::Used(bytes) = process.used_gpu_memory {
                        set(process.pid, gpus.len(), bytes);
                    }
                }
                gpus.push(Gpu { name, vram: Total { used: memory.used, total: memory.total } });
            }
        }
        // AMD's driver tells in sysfs, and in the fdinfo of the device files that processes have open.
        let amd = amd_gpus();
        if !amd.is_empty() {
            for pid in self.system.processes().keys().map(|pid| pid.as_u32()) {
                for (pdev, bytes) in drm_vram(pid) {
                    if let Some(i) = amd.iter().position(|(gpu_pdev, _)| *gpu_pdev == pdev) {
                        set(pid, gpus.len() + i, bytes);
                    }
                }
            }
        }
        gpus.extend(amd.into_iter().map(|(_, gpu)| gpu));
        (gpus, vram)
    }

    #[cfg(not(target_os = "linux"))]
    fn gpus(&self) -> (Vec<Gpu>, HashMap<u32, Vec<u64>>) {
        Default::default()
    }
}

/// The RAM, and the RAM plus swap, that only the process uses: its anonymous memory.
#[cfg(target_os = "linux")]
fn private_memory(pid: u32) -> Option<(u64, u64)> {
    let status = fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    let ram = parse_size(field(&status, "RssAnon")?)?;
    Some((ram, ram + field(&status, "VmSwap").and_then(parse_size).unwrap_or(0)))
}

#[cfg(not(target_os = "linux"))]
fn private_memory(_pid: u32) -> Option<(u64, u64)> {
    None
}

/// AMD GPUs by PCI address, like "0000:03:00.0", with their VRAM use.
#[cfg(target_os = "linux")]
fn amd_gpus() -> Vec<(String, Gpu)> {
    let Ok(cards) = fs::read_dir("/sys/class/drm") else { return Vec::new() };
    let mut gpus: Vec<_> = cards
        .flatten()
        // Leaves out the connectors, like "card1-DP-1".
        .filter(|card| card.file_name().to_str().is_some_and(|name| name.starts_with("card") && !name.contains('-')))
        .filter_map(|card| {
            let device = card.path().join("device");
            let read = |file: &str| -> Option<u64> { fs::read_to_string(device.join(file)).ok()?.trim().parse().ok() };
            let (total, used) = (read("mem_info_vram_total")?, read("mem_info_vram_used")?);
            let pdev = fs::canonicalize(&device).ok()?.file_name()?.to_string_lossy().into_owned();
            Some((pdev, Gpu { name: format!("AMD {}", card.file_name().to_string_lossy()), vram: Total { used, total } }))
        })
        .collect();
    gpus.sort_by(|a, b| a.1.name.cmp(&b.1.name));
    gpus
}

/// The VRAM that the DRM clients of a process use, by the PCI address of the GPU.
#[cfg(target_os = "linux")]
fn drm_vram(pid: u32) -> HashMap<String, u64> {
    let mut vram = HashMap::new();
    // Only readable for the user's own processes, which are the ones it can end anyway.
    let Ok(fds) = fs::read_dir(format!("/proc/{pid}/fd")) else { return vram };
    let mut clients = HashSet::new();
    for fd in fds.flatten() {
        if !fs::read_link(fd.path()).is_ok_and(|target| target.starts_with("/dev/dri/")) {
            continue;
        }
        let Ok(info) = fs::read_to_string(format!("/proc/{pid}/fdinfo/{}", fd.file_name().to_string_lossy())) else { continue };
        // Duplicated file descriptors share their client.
        if let Some((pdev, client, bytes)) = drm_client(&info)
            && clients.insert((pdev.to_owned(), client.to_owned()))
        {
            *vram.entry(pdev.to_owned()).or_default() += bytes;
        }
    }
    vram
}

/// The GPU's PCI address, the client id and the VRAM use in the fdinfo of a DRM device file.
#[cfg(target_os = "linux")]
fn drm_client(info: &str) -> Option<(&str, &str, u64)> {
    // Older kernels only have the legacy key.
    let vram = field(info, "drm-resident-vram").or_else(|| field(info, "drm-memory-vram")).and_then(parse_size).unwrap_or(0);
    Some((field(info, "drm-pdev")?, field(info, "drm-client-id")?, vram))
}

/// The value of a "key: value" line, as in the files of /proc.
#[cfg(target_os = "linux")]
fn field<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    text.lines().find_map(|line| line.strip_prefix(key)?.strip_prefix(':')).map(str::trim)
}

/// Sizes like "1234 kB" or "12 MiB".
#[cfg(target_os = "linux")]
fn parse_size(text: &str) -> Option<u64> {
    let mut words = text.split_whitespace();
    let number: u64 = words.next()?.parse().ok()?;
    let unit = match words.next() {
        None => 1,
        Some("kB" | "KiB") => 1 << 10,
        Some("MiB") => 1 << 20,
        Some("GiB") => 1 << 30,
        Some(_) => return None,
    };
    Some(number * unit)
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    #[test]
    fn amdgpu_fdinfo() {
        let info = "pos:\t0\nflags:\t02100002\ndrm-driver:\tamdgpu\ndrm-client-id:\t42\ndrm-pdev:\t0000:03:00.0\ndrm-total-vram:\t131072 KiB\ndrm-resident-vram:\t65536 KiB\ndrm-memory-vram:\t65536 KiB\ndrm-memory-gtt: \t2048 KiB\n";
        assert_eq!(drm_client(info), Some(("0000:03:00.0", "42", 64 << 20)));
        let legacy = "drm-driver:\tamdgpu\ndrm-pdev:\t0000:03:00.0\ndrm-client-id:\t7\ndrm-memory-vram:\t2048 KiB\n";
        assert_eq!(drm_client(legacy), Some(("0000:03:00.0", "7", 2 << 20)));
        assert_eq!(drm_client("pos:\t0\nflags:\t02\n"), None);
    }

    #[test]
    fn proc_status() {
        let status = "Name:\tbash\nVmRSS:\t    5000 kB\nRssAnon:\t    1200 kB\nRssFile:\t    3800 kB\nVmSwap:\t     300 kB\n";
        assert_eq!(field(status, "RssAnon").and_then(parse_size), Some(1200 << 10));
        assert_eq!(field(status, "VmSwap").and_then(parse_size), Some(300 << 10));
        assert_eq!(field(status, "Rss"), None);
    }
}
