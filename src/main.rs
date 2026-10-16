#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

#[cfg_attr(windows, path = "sampler_windows.rs")]
#[cfg_attr(not(windows), path = "sampler_unix.rs")]
mod sampler;

use eframe::egui;
use egui_extras::{Column, TableBuilder};
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    sync::mpsc,
    time::Duration,
};

const REFRESH: Duration = Duration::from_secs(1);
const NAME_TIP: &str = "Processes grouped by executable. Click a program with several processes to list them.";
const RAM_TIP: &str = "Physical memory that only this program uses, which ending it frees.";
const COMMIT_TIP: &str = "Memory that only this program uses, also counting what was moved out of RAM to the page file, swap or compressed memory. Its total is limited to RAM plus page file or swap.";
const CPU_TIP: &str = "Share of the time of all CPU cores, over the last second.";
const DISK_TIP: &str = "Bytes per second this program read from and wrote to disk, over the last second.";
const NET_TIP: &str = "Bytes per second this program sent and received, over the last second. Needs admin rights.";

/// What a process uses, or all the processes of a program.
#[derive(Default)]
pub struct Usage {
    /// Physical memory that only the process uses.
    pub ram: u64,
    /// Memory that only the process uses, also counting what was moved out of RAM.
    pub commit: u64,
    /// Dedicated video memory used on each GPU of `Sample::gpus`, missing ones being zero.
    pub vram: Vec<u64>,
    /// Share of the time of all CPU cores since the previous sample, in percent.
    pub cpu: f32,
    /// Bytes per second read from and written to disk.
    pub disk: u64,
    /// Bytes per second sent and received over the network.
    pub net: u64,
}

impl Usage {
    fn add(&mut self, other: &Usage) {
        self.ram += other.ram;
        self.commit += other.commit;
        self.vram.resize(self.vram.len().max(other.vram.len()), 0);
        for (sum, vram) in self.vram.iter_mut().zip(&other.vram) {
            *sum += vram;
        }
        self.cpu += other.cpu;
        self.disk += other.disk;
        self.net += other.net;
    }
}

pub struct Proc {
    pub pid: u32,
    /// The executable's file name.
    pub name: String,
    pub usage: Usage,
}

/// How much of some memory is in use.
#[derive(Default)]
pub struct Total {
    pub used: u64,
    pub total: u64,
}

pub struct Gpu {
    pub name: String,
    pub vram: Total,
}

#[derive(Default)]
pub struct Sample {
    pub ram: Total,
    /// The commit charge against its limit, or RAM plus swap.
    pub commit: Total,
    pub gpus: Vec<Gpu>,
    pub procs: Vec<Proc>,
    /// Bytes per second over all network interfaces.
    pub net: u64,
    /// Whether the processes carry their own network use, which needs admin rights.
    pub net_per_process: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum Col {
    Name,
    Ram,
    Commit,
    Vram(usize),
    Cpu,
    Disk,
    Net,
}

impl Col {
    fn value(self, usage: &Usage) -> f64 {
        match self {
            Col::Name => 0.0,
            Col::Ram => usage.ram as f64,
            Col::Commit => usage.commit as f64,
            Col::Vram(gpu) => usage.vram.get(gpu).copied().unwrap_or(0) as f64,
            // As shown, for rows that show no CPU use to sort by name.
            Col::Cpu => (usage.cpu as f64 * 10.0).round() / 10.0,
            Col::Disk => usage.disk as f64,
            Col::Net => usage.net as f64,
        }
    }

    /// Nothing for zero, for the rest to stand out.
    fn text(self, usage: &Usage) -> String {
        let value = self.value(usage);
        match self {
            _ if value <= 0.0 => String::new(),
            Col::Cpu => format!("{value:.1}%"),
            Col::Disk | Col::Net => rate(value as u64),
            _ => size(value as u64),
        }
    }

    /// The system-wide use of the memory in the column.
    fn total(self, sample: &Sample) -> Option<&Total> {
        match self {
            Col::Ram => Some(&sample.ram),
            Col::Commit => Some(&sample.commit),
            Col::Vram(gpu) => sample.gpus.get(gpu).map(|gpu| &gpu.vram),
            _ => None,
        }
    }
}

/// The processes of a program, as those with the same executable.
struct Group<'a> {
    name: &'a str,
    usage: Usage,
    procs: Vec<&'a Proc>,
}

enum Row<'a> {
    Group(&'a Group<'a>),
    Proc(&'a Proc),
}

struct App {
    rx: mpsc::Receiver<Sample>,
    sample: Sample,
    sort: Col,
    descending: bool,
    filter: String,
    /// The programs with their processes listed below them.
    expanded: HashSet<String>,
    /// Whether to offer relaunching as admin, which per-process network needs.
    offer_admin: bool,
}

impl App {
    fn new(cc: &eframe::CreationContext) -> Self {
        let (tx, rx) = mpsc::channel();
        let ctx = cc.egui_ctx.clone();
        // Sampling takes a moment, which would stall the UI.
        std::thread::spawn(move || {
            let mut sampler = sampler::Sampler::new();
            while tx.send(sampler.sample()).is_ok() {
                ctx.request_repaint();
                std::thread::sleep(REFRESH);
            }
        });
        Self { rx, sample: Sample::default(), sort: Col::Ram, descending: true, filter: String::new(), expanded: HashSet::new(), offer_admin: sampler::offer_admin() }
    }

    /// By the sorted column only, the caller breaks ties for rows to stay in place between samples.
    fn order(&self, (a_name, a): (&str, &Usage), (b_name, b): (&str, &Usage)) -> Ordering {
        let order = match self.sort {
            Col::Name => by_name(a_name, b_name),
            col => col.value(a).total_cmp(&col.value(b)),
        };
        if self.descending {
            order.reverse()
        } else {
            order
        }
    }

    /// Only borrows the processes, for the filter to stay editable while the groups are shown.
    fn groups<'a>(&self, procs: &'a [Proc]) -> Vec<Group<'a>> {
        let filter = self.filter.to_lowercase();
        let mut groups = HashMap::new();
        for proc in procs.iter().filter(|proc| proc.name.to_lowercase().contains(&filter)) {
            let group = groups.entry(proc.name.as_str()).or_insert_with(|| Group { name: &proc.name, usage: Usage::default(), procs: Vec::new() });
            group.usage.add(&proc.usage);
            group.procs.push(proc);
        }
        let mut groups: Vec<_> = groups.into_values().collect();
        for group in &mut groups {
            group.procs.sort_by(|a, b| self.order((&a.name, &a.usage), (&b.name, &b.usage)).then(a.pid.cmp(&b.pid)));
        }
        groups.sort_by(|a, b| self.order((a.name, &a.usage), (b.name, &b.usage)).then_with(|| by_name(a.name, b.name)));
        groups
    }

    fn table_ui(&mut self, ui: &mut egui::Ui) {
        let gpus = &self.sample.gpus;
        let mut cols = vec![
            (Col::Name, "Program".to_owned(), NAME_TIP.to_owned()),
            (Col::Ram, "RAM".to_owned(), RAM_TIP.to_owned()),
            (Col::Commit, "Commit".to_owned(), COMMIT_TIP.to_owned()),
        ];
        for (i, gpu) in gpus.iter().enumerate() {
            cols.push((Col::Vram(i), vram_title(gpus, i), format!("Dedicated video memory that this program uses on the {}.", gpu.name)));
        }
        cols.push((Col::Cpu, "CPU".to_owned(), CPU_TIP.to_owned()));
        cols.push((Col::Disk, "Disk".to_owned(), DISK_TIP.to_owned()));
        cols.push((Col::Net, "Network".to_owned(), NET_TIP.to_owned()));

        let groups = self.groups(&self.sample.procs);
        let mut rows = Vec::new();
        for group in &groups {
            rows.push(Row::Group(group));
            if group.procs.len() > 1 && self.expanded.contains(group.name) {
                rows.extend(group.procs.iter().map(|&proc| Row::Proc(proc)));
            }
        }
        let cpu: f32 = self.sample.procs.iter().map(|proc| proc.usage.cpu).sum();
        let disk: u64 = self.sample.procs.iter().map(|proc| proc.usage.disk).sum();

        let (mut sort, mut toggle, mut relaunch) = (None, None, false);
        let height = ui.spacing().interact_size.y;
        // The titles, with the filter or the system-wide totals below them.
        let header_height = 2.0 * (height + ui.spacing().item_spacing.y);
        let right = egui::Layout::right_to_left(egui::Align::Center);
        // Selectable text would take the clicks on the rows.
        ui.style_mut().interaction.selectable_labels = false;
        // A floating scroll bar would cover the last column.
        ui.spacing_mut().scroll = egui::style::ScrollStyle::solid();
        TableBuilder::new(ui)
            .striped(true)
            .resizable(true)
            .sense(egui::Sense::click())
            // Cells laid out right to left grow their click area over the previous column, hiding its header button,
            // so only their contents are right-aligned.
            .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
            // Only a remainder column that isn't resizable keeps filling the table as the window resizes.
            .column(Column::remainder().at_least(150.0).clip(true).resizable(false))
            .columns(Column::initial(95.0).at_least(50.0), cols.len() - 1)
            .header(header_height, |mut header| {
                for (col, title, tip) in &cols {
                    header.col(|ui| {
                        let arrow = match (self.sort == *col, self.descending) {
                            (false, _) => "",
                            (true, true) => " ⏷",
                            (true, false) => " ⏶",
                        };
                        let align = if *col == Col::Name { egui::Align::Min } else { egui::Align::Max };
                        ui.with_layout(egui::Layout::top_down(align), |ui| {
                            if ui.selectable_label(self.sort == *col, format!("{title}{arrow}")).on_hover_text(tip).clicked() {
                                sort = Some(*col);
                            }
                            match col {
                                // The filter, with the button to see per-process network where the room is.
                                Col::Name => {
                                    ui.with_layout(right, |ui| {
                                        if self.offer_admin && ui.button("Run as admin").on_hover_text("Relaunch with admin rights to also see each program's network use").clicked() {
                                            relaunch = true;
                                        }
                                        ui.add(egui::TextEdit::singleline(&mut self.filter).hint_text("Filter programs").desired_width(ui.available_width()));
                                    });
                                }
                                Col::Cpu => {
                                    bar(ui, cpu / 100.0, format!("{cpu:.0}%"));
                                }
                                // Rates have no capacity to fill a bar, so just the figure.
                                Col::Disk => {
                                    ui.label(rate(disk));
                                }
                                Col::Net => {
                                    ui.label(rate(self.sample.net));
                                }
                                _ => {
                                    if let Some(total) = col.total(&self.sample).filter(|total| total.total > 0) {
                                        bar(ui, total.used as f32 / total.total as f32, used_of(total));
                                    }
                                }
                            }
                        });
                    });
                }
            })
            .body(|body| {
                body.rows(height, rows.len(), |mut row| {
                    let index = row.index();
                    let (usage, group) = match rows[index] {
                        Row::Group(group) => (&group.usage, Some(group)),
                        Row::Proc(proc) => (&proc.usage, None),
                    };
                    row.col(|ui| match rows[index] {
                        Row::Group(group) => {
                            let several = group.procs.len() > 1;
                            let arrow = match (several, self.expanded.contains(group.name)) {
                                (false, _) => "",
                                (true, false) => "⏵",
                                (true, true) => "⏷",
                            };
                            ui.add_sized([12.0, height], egui::Label::new(arrow));
                            let text = if several { format!("{} ({})", group.name, group.procs.len()) } else { group.name.to_owned() };
                            ui.add(egui::Label::new(text).truncate());
                        }
                        Row::Proc(proc) => {
                            ui.add_space(32.0);
                            ui.label(egui::RichText::new(format!("PID {}", proc.pid)).weak());
                        }
                    });
                    for (col, ..) in &cols[1..] {
                        row.col(|ui| {
                            ui.with_layout(right, |ui| ui.label(col.text(usage)));
                        });
                    }
                    if let Some(group) = group
                        && group.procs.len() > 1
                        && row.response().clicked()
                    {
                        toggle = Some(group.name.to_owned());
                    }
                });
            });

        if let Some(col) = sort {
            // Names read best from A to Z, amounts from the biggest.
            self.descending = if col == self.sort { !self.descending } else { col != Col::Name };
            self.sort = col;
        }
        if let Some(name) = toggle
            && !self.expanded.remove(&name)
        {
            self.expanded.insert(name);
        }
        if relaunch {
            sampler::relaunch_as_admin();
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }
}

impl eframe::App for App {
    fn logic(&mut self, _ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Some(sample) = self.rx.try_iter().last() {
            self.sample = sample;
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ui, |ui| self.table_ui(ui));
    }
}

/// Ignoring case, unless that's the only difference.
fn by_name(a: &str, b: &str) -> Ordering {
    a.to_lowercase().cmp(&b.to_lowercase()).then_with(|| a.cmp(b))
}

/// Like "0.3 MB", "512 MB" or "3.2 GB".
fn size(bytes: u64) -> String {
    let (unit, decimals, name) = unit(bytes);
    format!("{:.*} {name}", decimals, bytes as f64 / unit)
}

/// Like "18.2 / 31.2 GB", both in the unit that suits the total.
fn used_of(total: &Total) -> String {
    let (unit, decimals, name) = unit(total.total);
    format!("{:.*} / {:.*} {name}", decimals, total.used as f64 / unit, decimals, total.total as f64 / unit)
}

/// The unit that suits a size, how many decimals to show it with, and the unit's name.
fn unit(bytes: u64) -> (f64, usize, &'static str) {
    const MB: f64 = (1 << 20) as f64;
    match bytes as f64 / MB {
        mb if mb < 10.0 => (MB, 1, "MB"),
        mb if mb < 1000.0 => (MB, 0, "MB"),
        _ => (MB * 1024.0, 1, "GB"),
    }
}

/// A transfer speed like "340 KB/s" or "12 MB/s", down to the kilobyte since idle rates are small.
fn rate(bytes_per_sec: u64) -> String {
    let kb = bytes_per_sec as f64 / 1024.0;
    if kb < 1.0 {
        String::new()
    } else if kb < 1000.0 {
        format!("{kb:.0} KB/s")
    } else if kb < 1000.0 * 1024.0 {
        format!("{:.1} MB/s", kb / 1024.0)
    } else {
        format!("{:.1} GB/s", kb / 1024.0 / 1024.0)
    }
}

/// "VRAM", followed by the GPU's vendor when there are several, like on laptops with integrated graphics.
fn vram_title(gpus: &[Gpu], gpu: usize) -> String {
    if gpus.len() == 1 {
        return "VRAM".to_owned();
    }
    let vendor = gpus[gpu].name.split_whitespace().next().unwrap_or_default();
    format!("VRAM {}", vendor.replace("(R)", "").replace("(TM)", ""))
}

/// A bar with its text inside, filling the column, which widens if the text needs it.
fn bar(ui: &mut egui::Ui, fraction: f32, text: String) -> egui::Response {
    let font = egui::TextStyle::Button.resolve(ui.style());
    let width = ui.painter().layout_no_wrap(text.clone(), font, egui::Color32::PLACEHOLDER).size().x + 2.0 * ui.spacing().item_spacing.x;
    ui.add(egui::ProgressBar::new(fraction).text(text).desired_width(width.max(ui.available_width())))
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([950.0, 750.0]),
        ..Default::default()
    };
    eframe::run_native("tuskManager", options, Box::new(|cc| Ok(Box::new(App::new(cc)))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_own_process() {
        let mut sampler = sampler::Sampler::new();
        sampler.sample();
        std::thread::sleep(Duration::from_millis(300));
        let sample = sampler.sample();
        let own = sample.procs.iter().find(|proc| proc.pid == std::process::id()).expect("own process listed");
        assert!(own.usage.ram > 0 && !own.name.is_empty());
        assert!(0 < sample.ram.used && sample.ram.used <= sample.ram.total);
        assert!(sample.commit.used <= sample.commit.total);
        let cpu: f32 = sample.procs.iter().map(|proc| proc.usage.cpu).sum();
        assert!(cpu <= 101.0, "total CPU {cpu}%");
    }
}
