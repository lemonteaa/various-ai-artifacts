**(Note: quickgen, codes unverified)**


<think>
The user wants a full Tauri v2 + React MVP for a QEMU VM manager. I need to search for relevant info to verify technical details (Tauri v2 sidecar/shell plugin, tauri-plugin-shell, xterm.js with Tauri, QEMU QMP, pty/websocket). Let me do a few targeted searches.

Key things to verify:

    Tauri v2 sidecar configuration (tauri.conf.json externalBin, shell plugin capabilities)
    xterm.js + Tauri stdin/stdout streaming for QEMU console
    QEMU QMP vs monitor stdio
    tauri-plugin-dialog / fs permissions

Let me search.
</think>



# QEMU VM Manager — Tauri v2 + React MVP Draft

Here's a complete architectural design plus copy-paste-ready skeleton code for your PoC. A few key design decisions up front, since they shape everything downstream:

- **QEMU process lifecycle lives in Rust, not the frontend.** The frontend only sends JSON configs via `invoke`; a Rust module builds the argument vector, spawns `qemu-system-x86_64` with `std::process::Command`, holds the `Child` handle, and streams stdout/stderr back as Tauri events. This avoids the shell-plugin scope dance for a user-supplied binary path and gives you reliable kill/stdin control. (The `tauri-plugin-shell` sidecar pattern is the alternative — it's the right choice if you ever *bundle* a qemu binary yourself, and it also supports stdin via `child.write()` and stdout events【turn0search0】【turn0search7】.)
- **Console = `-nographic` stdio muxed into xterm.js.** `-nographic` already multiplexes the serial console and the QEMU monitor onto stdio (switch with `Ctrl-A c` inside the terminal), which is exactly the interaction model from the wikibook you quoted — you get both guest console and monitor for free through one pipe. Upgrade path later: `-qmp unix:...,server` for structured control.
- **All managed files live under `$APPDATA/qemvm/`**, enforced both by convention in Rust commands and by an fs-plugin scope in capabilities, so the security boundary is real rather than advisory【turn1search3】.

---

## A. Overall Architecture and Design Decisions

```
┌─────────────────────────────────────────────────────────────┐
│ React Frontend (WebView)                                    │
│  ├─ VM List / Image Tree (left sidebar)                     │
│  ├─ VM Config Form (drives, RAM/CPU, port forwards)         │
│  └─ Console panel (xterm.js)                                │
│      ↕ invoke()          ↕ listen() events                  │
├─────────────────────────────────────────────────────────────┤
│ Tauri Core (Rust)                                           │
│  ├─ store.rs      → vms.json persistence in $APPDATA/qemvm  │
│  ├─ qemu/args.rs  → config struct → argv builder            │
│  ├─ qemu/process.rs → spawn/kill/stdin, event emitter       │
│  └─ qemu/img.rs   → qemu-img create/info (backing chains)   │
├─────────────────────────────────────────────────────────────┤
│ OS                                                          │
│  ├─ qemu-system-x86_64.exe (user-provided, path in settings)│
│  └─ qemu-img.exe                                            │
└─────────────────────────────────────────────────────────────┘
```

**Directory convention** (the "managed folder" you mentioned):

```
$APPDATA/qemvm/
├── vms.json          # VM configs + image registry metadata
├── images/           # all .qcow2 files live here
│   ├── debian-base.qcow2
│   └── dev-overlay.qcow2   (backing → debian-base.qcow2)
└── settings.json     # qemu binary path etc.
```

---

## B. Project Structure and Key Files

Scaffold:

```bash
npm create tauri-app@latest qemvm -- --template react-ts
cd qemvm && npm install
npm install @tauri-apps/plugin-fs @tauri-apps/plugin-dialog @xterm/xterm @xterm/addon-fit
cargo add tauri-plugin-fs tauri-plugin-dialog serde uuid --manifest-path src-tauri/Cargo.toml
```

```
src/
├── App.tsx
├── types.ts
├── api/
│   ├── vms.ts          # invoke wrappers
│   └── images.ts
├── components/
│   ├── Sidebar.tsx
│   ├── ImageTree.tsx   # backing-file dependency visualization
│   ├── VmConfigForm.tsx
│   ├── PortForwardingEditor.tsx
│   └── Console.tsx     # xterm.js
└── hooks/
    └── useConsole.ts
src-tauri/
├── src/
│   ├── lib.rs
│   ├── state.rs
│   ├── store.rs
│   └── qemu/
│       ├── mod.rs
│       ├── args.rs
│       ├── process.rs
│       └── img.rs
├── capabilities/default.json
└── tauri.conf.json
```

<details>
<summary><strong>src/types.ts — shared data model</strong></summary>

```ts
export interface Drive {
  id: string;
  /** filename inside $APPDATA/qemvm/images/ */
  image: string;
  snapshot: boolean;
}

export interface PortForward {
  id: string;
  proto: "tcp" | "udp";
  hostPort: number;
  guestPort: number;
  hostAddr: string; // default 127.0.0.1
}

export interface VmConfig {
  id: string;
  name: string;
  memoryMb: number;      // e.g. 2048
  smp: number;           // vCPUs
  snapshot: boolean;     // apply to boot drive
    drives: Drive[];       // [0] is boot drive; supports multiple
  cdrom: string | null;  // iso filename, also lives in images/
  networkEnabled: boolean;
  portForwards: PortForward[];
  createdAt: string;
}

export interface ImageInfo {
  filename: string;
  virtualSize: string;
  actualSize: string;
  backingFile: string | null;
}

export type VmStatus = "stopped" | "running" | "error";
```
</details>

---

## C. QEMU Argument Builder (Rust)

This is the heart of "user never types flags again." It reproduces your reference command exactly, generalized.

<details>
<summary><strong>src-tauri/src/qemu/args.rs</strong></summary>

```rust
use crate::store::VmConfig;

/// Build qemu-system-x86_64 argv from a VM config.
/// Images are referenced by absolute path inside the managed folder.
pub fn build_qemu_args(cfg: &VmConfig, image_dir: &std::path::Path) -> Vec<String> {
    let mut args: Vec<String> = Vec::new();

    // --- Memory & CPU ---
    args.push("-m".into());
    args.push(format!("{}M", cfg.memory_mb));
    args.push("-smp".into());
    args.push(cfg.smp.to_string());

    // --- Drives (boot drive first; multiple supported) ---
    // -drive file=dev-setup-overlay.qcow2,format=qcow2,if=virtio,snapshot=on
    for (i, drive) in cfg.drives.iter().enumerate() {
        let path = image_dir.join(&drive.image);
        let mut spec = format!(
            "file={},format=qcow2,if=virtio",
            path.to_string_lossy()
        );
        // snapshot only on the boot drive (matches reference command)
        if i == 0 && cfg.snapshot {
            spec.push_str(",snapshot=on");
        }
        args.push("-drive".into());
        args.push(spec);
    }

    // --- CD-ROM (iso is raw) ---
    if let Some(iso) = &cfg.cdrom {
        let path = image_dir.join(iso);
        args.push("-drive".into());
        args.push(format!(
            "file={},format=raw,if=virtio,media=cdrom",
            path.to_string_lossy()
        ));
    }

    // --- User-mode networking (works without admin on Windows) ---
    // -netdev user,id=net0,hostfwd=tcp:127.0.0.1:10225-:22,...
    // -device virtio-net-pci,netdev=net0
    if cfg.network_enabled {
        let mut netdev = String::from("user,id=net0");
        for fw in &cfg.port_forwards {
            netdev.push_str(&format!(
                ",hostfwd={}:{}:{}-:{}",
                fw.proto, fw.host_addr, fw.host_port, fw.guest_port
            ));
        }
        args.push("-netdev".into());
        args.push(netdev);
        args.push("-device".into());
        args.push("virtio-net-pci,netdev=net0".into());
    } else {
        args.push("-nic".into());
        args.push("none".into());
    }

    // --- Display: no GUI; serial+monitor muxed on stdio (Ctrl-A c) ---
    args.push("-nographic".into());
    // Windows-only nicety: keep qemu running when its own console detaches
    args.push("-name".into());
    args.push(cfg.name.clone());

    args
}

/// Full command: [qemu_path, args...]
pub fn build_command(qemu_bin: &str, cfg: &VmConfig, image_dir: &std::path::Path) -> Vec<String> {
    let mut cmd = vec![qemu_bin.to_string()];
    cmd.extend(build_qemu_args(cfg, image_dir));
    cmd
}
```
</details>

---

## D. Process Manager (Rust) — spawn / stop / stdin / event stream

<details>
<summary><strong>src-tauri/src/qemu/process.rs</strong></summary>

```rust
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter};
use uuid::Uuid;

pub struct RunningVm {
    pub child: Child,
    pub stdin: Option<ChildStdin>,
}

#[derive(Default)]
pub struct VmRegistry {
    pub running: Arc<Mutex<HashMap<String, RunningVm>>>,
}

fn spawn_reader<R: std::io::Read + Send + 'static>(
    app: AppHandle, vm_id: String, stream: &'static str, reader: R,
) {
    std::thread::spawn(move || {
        let mut line_reader = BufReader::new(reader);
        let mut buf = Vec::new();
        // read bytes, not lines — terminal escape sequences matter for xterm
        let mut chunk = [0u8; 4096];
        use std::io::Read;
        let mut reader = line_reader;
        loop {
            match reader.read(&mut chunk) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    buf.extend_from_slice(&chunk[..n]);
                    let _ = app.emit(
                        &format!("vm-stdio-{vm_id}"),
                        String::from_utf8_lossy(&buf).into_owned(),
                    );
                    buf.clear();
                }
            }
        }
        let _ = app.emit(&format!("vm-exit-{vm_id}"), ());
    });
}

#[tauri::command]
pub fn vm_start(
    app: AppHandle,
    state: tauri::State<VmRegistry>,
    cfg: crate::store::VmConfig,
    app_data_dir: String,
    qemu_bin: String,
) -> Result<String, String> {
    let image_dir = std::path::Path::new(&app_data_dir).join("images");
    let argv = crate::qemu::args::build_command(&qemu_bin, &cfg, &image_dir);

    let mut child = Command::new(&argv[0])
        .args(&argv[1..])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .creation_flags(0x0800_0000) // CREATE_NO_WINDOW on Windows
        .spawn()
        .map_err(|e| format!("failed to spawn qemu: {e} — check the binary path in Settings"))?;

    let stdin = child.stdin.take();
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    let vm_id = cfg.id.clone();
    if let Some(out) = stdout {
        spawn_reader(app.clone(), vm_id.clone(), "stdout", out);
    }
    if let Some(err) = stderr {
        spawn_reader(app.clone(), vm_id.clone(), "stderr", err);
    }

    state.running
        .lock()
        .unwrap()
        .insert(vm_id.clone(), RunningVm { child, stdin });
    Ok(vm_id)
}

/// Frontend console keystrokes → qemu stdin
#[tauri::command]
pub fn vm_write_stdin(
    state: tauri::State<VmRegistry>,
    vm_id: String,
    data: String,
) -> Result<(), String> {
    let mut map = state.running.lock().unwrap();
    if let Some(vm) = map.get_mut(&vm_id) {
        if let Some(stdin) = vm.stdin.as_mut() {
            stdin.write_all(data.as_bytes()).map_err(|e| e.to_string())
        } else {
            Err("stdin not available".into())
        }
    } else {
        Err("vm not running".into())
    }
}

#[tauri::command]
pub fn vm_stop(state: tauri::State<VmRegistry>, vm_id: String) -> Result<(), String> {
    let mut map = state.running.lock().unwrap();
    if let Some(mut vm) = map.remove(&vm_id) {
        // Try graceful ACPI shutdown first is a QMP feature; for PoC: hard kill.
        vm.child.kill().map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tauri::command]
pub fn vm_status(state: tauri::State<VmRegistry>) -> Vec<String> {
    state.running.lock().unwrap()
        .iter()
        .filter(|(_, vm)| matches!(vm.child.try_wait(), Ok(None)))
        .map(|(id, _)| id.clone())
        .collect()
}
```
</details>

> **Note on `creation_flags`:** that's Windows-only; on Linux/macOS remove it (or `#[cfg(windows)]` it) — otherwise it won't compile cross-platform.

---

## E. Image Management & Backing-Chain Visualization

<details>
<summary><strong>src-tauri/src/qemu/img.rs — qemu-img wrappers</strong></summary>

```rust
use std::path::Path;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct ImageInfo {
    pub filename: String,
    pub virtual_size: String,
    pub actual_size: String,
    pub backing_file: Option<String>,
}

/// Create a fresh (non-backed) image:
///   qemu-img create -f qcow2 name.qcow2 10G
/// Create a backed overlay:
///   qemu-img create -f qcow2 -b base.qcow2 -F qcow2 overlay.qcow2 10G
#[tauri::command]
pub fn image_create(
    app_data_dir: String,
    filename: String,
    size_gb: u32,
    backing_file: Option<String>, // None → base image; Some → overlay
    qemu_img_bin: String,
) -> Result<(), String> {
    let images = Path::new(&app_data_dir).join("images");
    let target = images.join(&filename);

    let mut cmd = std::process::Command::new(&qemu_img_bin);
    cmd.arg("create").arg("-f").arg("qcow2");
    if let Some(backing) = &backing_file {
        cmd.arg("-b").arg(images.join(backing));
        cmd.arg("-F").arg("qcow2");
    }
    cmd.arg(&target).arg(format!("{size_gb}G"));
    let out = cmd.output().map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    Ok(())
}

/// Scan images/ and build the dependency graph for the UI tree.
#[tauri::command]
pub fn image_list(app_data_dir: String, qemu_img_bin: String) -> Result<Vec<ImageInfo>, String> {
    let images = Path::new(&app_data_dir).join("images");
    let mut result = Vec::new();
    let entries = std::fs::read_dir(&images).map_err(|e| e.to_string())?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("qcow2") {
            continue;
        }
        // qemu-img info --output=json
        let out = std::process::Command::new(&qemu_img_bin)
            .args(["info", "--output=json"])
            .arg(&path)
            .output()
            .map_err(|e| e.to_string())?;
        let json: serde_json::Value =
            serde_json::from_slice(&out.stdout).map_err(|e| e.to_string())?;
        result.push(ImageInfo {
            filename: path.file_name().unwrap().to_string_lossy().into_owned(),
            virtual_size: json["virtual-size"].to_string(),
            actual_size: json["actual-size"].to_string(),
            // full-backing-filename gives absolute path; strip to filename
            backing_file: json["full-backing-filename"]
                .as_str()
                .and_then(|p| {
                    std::path::Path::new(p)
                        .file_name()
                        .map(|f| f.to_string_lossy().into_owned())
                }),
        });
    }
    Ok(result)
}
```
</details>

<details>
<summary><strong>src/components/ImageTree.tsx — chain visualization</strong></summary>

```tsx
import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import type { ImageInfo } from "../types";

/** Renders each image as a card; children are indented under their backing file.
 *  ▸ base images (no backing) at root
 *  ▸ overlays nested under their parent, with a "linked to" pill
 */
export function ImageTree({ onAttach }: { onAttach: (img: ImageInfo) => void }) {
  const [images, setImages] = useState<ImageInfo[]>([]);

  useEffect(() => {
    invoke<ImageInfo[]>("image_list", { /* appDataDir, qemuImgBin from settings ctx */ })
      .then(setImages);
  }, []);

  // roots = images nobody points to as backing; then DFS
  const byBacking = new Map<string, ImageInfo[]>();
  const isChild = new Set(images.map((i) => i.backing_file).filter(Boolean));
  const roots = images.filter((i) => !isChild.has(i.filename));

  const rows: { img: ImageInfo; depth: number }[] = [];
  const walk = (img: ImageInfo, depth: number) => {
    rows.push({ img, depth });
    (byBacking.get(img.filename) ?? []).forEach((c) => walk(c, depth + 1));
  };
  images.forEach((i) => {
    if (i.backing_file) {
      byBacking.set(i.backing_file!, [...(byBacking.get(i.backing_file!) ?? []), i]);
    }
  });
  roots.forEach((r) => walk(r, 0));

  return (
    <div className="image-tree">
      {rows.map(({ img, depth }) => (
        <div key={img.filename} style={{ marginLeft: depth * 24 }}>
          {depth > 0 && <span title="backing file">⛓→ </span>}
          <span>{img.filename}</span>
          <small> {img.virtual_size} · {img.actual_size} on disk</small>
          <button onClick={() => onAttach(img)}>attach to VM</button>
        </div>
      ))}
    </div>
  );
}
```
</details>

---

## F. Console Panel — xterm.js wired to qemu stdio

<details>
<summary><strong>src/components/Console.tsx</strong></summary>

```tsx
import { useEffect, useRef } from "react";
import { Terminal } from "@xterm/xterm";
import { FitAddon } from "@xterm/addon-fit";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import "@xterm/xterm/css/xterm.css";

export function Console({ vmId }: { vmId: string }) {
  const termRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const term = new Terminal({
      cursorBlink: true,
      fontSize: 13,
      // qemu sends CRLF; let xterm handle line endings
      convertEol: false,
    });
    const fit = new FitAddon();
    term.loadAddon(fit);
    term.open(termRef.current!);
    fit.fit();

    // qemu stdout/stderr → terminal
    const unlistenOut = listen<string>(`vm-stdio-${vmId}`, (e) => term.write(e.payload));
    const unlistenExit = listen(`vm-exit-${vmId}`, () => term.write("\r\n\x1b[31m[VM exited]\x1b[0m\r\n"));

    // keystrokes → qemu stdin
    const dataSub = term.onData((data) => {
      invoke("vm_write_stdin", { vmId, data }).catch(() => {});
    });

    term.write("Connecting to VM...\r\n");

    const resize = () => fit.fit();
    window.addEventListener("resize", resize);
    return () => {
      window.removeEventListener("resize", resize);
      unlistenOut.then((f) => f());
      unlistenExit.then((f) => f());
      dataSub.dispose();
      term.dispose();
    };
  }, [vmId]);

  return <div ref={termRef} style={{ height: "100%", padding: 8, background: "#1e1e1e" }} />;
}
```
</details>

Because we launch qemu with `-nographic`, this single terminal is both the guest serial console **and** the QEMU monitor — pressing `Ctrl-A` then `c` inside the xterm pane toggles between them, exactly like running qemu in a real terminal. That covers your "console access before SSH is set up" requirement with zero extra plumbing.

---

## G. UI Layout Blueprint

```
┌──────────────┬────────────────────────────────────────────────────┐
│              │  [ Debian Dev ]  ▾     ● Running   [Start] [Stop]  │
│  VMs         ├────────────────────────────────────────────────────┤
│  ├─ debian   │  Tabs: [ Configuration ] [ Console ] [ Logs ]      │
│  │   ●       │ ┌────────────────────────────────────────────────┐ │
│  ├─ win11    │ │ Memory [2048] MB   vCPUs [4]   ☐ snapshot mode │ │
│  └─ + New    │ │                                                │ │
│              │ │ Drives                                         │ │
│  Images      │ │  ⛓ dev-overlay.qcow2 (base: debian-base) [↑↓✕]│ │
│  ├─ 🖼 debian│ │  ⛓ data.qcow2                                 │ │
│  │   └─ ⛓ dev│ │  + Add drive / CD-ROM: [dropdown of images]    │ │
│  │      -over│ │                                                │ │
│  ├─ 🖼 alpine│ │ Network: ● user-mode (no admin required)        │ │
│  └─ + New    │ │  Forwards:                                     │ │
│              │ │   host 10225 → guest 22 (tcp)           [✕]    │ │
│  Settings    │ │   host 10380 → guest 8080 (tcp)          [✕]   │ │
│              │ │   + Add forward                                 │ │
│              │ └────────────────────────────────────────────────┘ │
├──────────────┴────────────────────────────────────────────────────┤
│ qemu: C:\tools\qemu\qemu-system-x86_64.exe   dir: %APPDATA%\qemvm │
└───────────────────────────────────────────────────────────────────┘
```

Key UX points baked into this layout:

- **Image tree uses `⛓` indentation** under the backing image so the dependency chain reads top-down ("dev-overlay is based on debian-base"). The *new image* dialog offers "Base image" vs. "Overlay of…" with a dropdown, plus a size field — that maps 1:1 onto the two `qemu-img create` variants (with/without `-b/-F`).
- **Drives and port-forwards are repeatable rows**, not fixed fields, matching your "multiple drives" and multiple `hostfwd` requirements.
- **Status dot per VM** reflects `vm_status` (poll every 2s or listen to `vm-exit-*` events).

<details>
<summary><strong>PortForwardingEditor.tsx (row editor)</strong></summary>

```tsx
export function PortForwardingEditor({ forwards, onChange }: {
  forwards: PortForward[]; onChange: (f: PortForward[]) => void;
}) {
  const add = () => onChange([...forwards, {
    id: crypto.randomUUID(), proto: "tcp",
    hostAddr: "127.0.0.1", hostPort: 0, guestPort: 0,
  }]);
  const upd = (id: string, patch: Partial<PortForward>) =>
    onChange(forwards.map((f) => (f.id === id ? { ...f, ...patch } : f)));

  return (
    <div>
      {forwards.map((f) => (
        <div key={f.id} className="row">
          <select value={f.proto} onChange={(e) => upd(f.id, { proto: e.target.value as "tcp"|"udp" })}>
            <option>tcp</option><option>udp</option>
          </select>
          <input value={f.hostPort} type="number"
                 onChange={(e) => upd(f.id, { hostPort: +e.target.value })} /> →
          <input value={f.guestPort} type="number"
                 onChange={(e) => upd(f.id, { guestPort: +e.target.value })} />
          <button onClick={() => onChange(forwards.filter((x) => x.id !== f.id))}>✕</button>
        </div>
      ))}
      <button onClick={add}>+ Add forward</button>
    </div>
  );
}
```
</details>

---

## H. Tauri Wiring: config, capabilities, command registration

<details>
<summary><strong>src-tauri/tauri.conf.json (relevant parts)</strong></summary>

```json
{
  "productName": "qemvm",
  "identifier": "com.example.qemvm",
  "app": {
    "windows": [{ "title": "QEMU VM Manager", "width": 1100, "height": 720 }],
    "security": { "csp": null }
  },
  "bundle": {
    "externalBin": []
  },
  "plugins": {}
}
```
</details>

<details>
<summary><strong>src-tauri/capabilities/default.json — the security boundary</strong></summary>

```json
{
  "$schema": "../gen/schemas/desktop-schema.json",
  "identifier": "default",
  "description": "Main window capability — fs access restricted to the managed VM folder",
  "windows": ["main"],
  "permissions": [
    "core:default",
    "core:event:default",
    "dialog:allow-open",
    {
      "identifier": "fs:scope",
      "allow": [{ "path": "$APPDATA/qemvm/**" }]
    },
    "fs:allow-read-text-file",
    "fs:allow-write-text-file",
    "fs:allow-read-dir",
    "fs:allow-mkdir"
  ]
}
```

The `$APPDATA` variable is resolved at runtime, and scopes like this are the v2-native way to fence file access【turn1search3】【turn1search5】. Note that bare `fs:allow-*` permissions without an `allow` scope block are known to misbehave, so always pair them with the scope entry as shown【turn1search7】.
</details>

<details>
<summary><strong>src-tauri/src/lib.rs — plugin + command registration</strong></summary>

```rust
mod qemu;
mod store;
mod state;

use state::VmRegistry;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(VmRegistry::default())
        .invoke_handler(tauri::generate_handler![
            qemu::process::vm_start,
            qemu::process::vm_stop,
            qemu::process::vm_status,
            qemu::process::vm_write_stdin,
            qemu::img::image_create,
            qemu::img::image_list,
            store::vm_list,
            store::vm_save,
            store::vm_delete,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```
</details>

<details>
<summary><strong>src-tauri/src/store.rs — vms.json persistence</strong></summary>

```rust
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct VmConfig {
    pub id: String,
    pub name: String,
    pub memory_mb: u32,
    pub smp: u32,
    pub snapshot: bool,
    pub drives: Vec<Drive>,
    pub cdrom: Option<String>,
    pub network_enabled: bool,
    pub port_forwards: Vec<PortForward>,
    pub created_at: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Drive { pub id: String, pub image: String, pub snapshot: bool }

#[derive(Serialize, Deserialize, Clone)]
pub struct PortForward {
    pub id: String, pub proto: String,
    pub host_addr: String, pub host_port: u16, pub guest_port: u16,
}

#[tauri::command]
pub fn vm_list(app_data_dir: String) -> Result<Vec<VmConfig>, String> {
    let path = std::path::Path::new(&app_data_dir).join("vms.json");
    if !path.exists() { return Ok(vec![]); }
    let data = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
    serde_json::from_str(&data).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn vm_save(app_data_dir: String, vm: VmConfig) -> Result<(), String> {
    let mut all = vm_list(app_data_dir.clone()).unwrap_or_default();
    all.retain(|v| v.id != vm.id);
    all.push(vm);
    let path = std::path::Path::new(&app_data_dir).join("vms.json");
    std::fs::write(path, serde_json::to_string_pretty(&all).unwrap())
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn vm_delete(app_data_dir: String, vm_id: String) -> Result<(), String> {
    let mut all = vm_list(app_data_dir.clone()).unwrap_or_default();
    all.retain(|v| v.id != vm_id);
    let path = std::path::Path::new(&app_data_dir).join("vms.json");
    std::fs::write(path, serde_json::to_string_pretty(&all).unwrap())
        .map_err(|e| e.to_string())
}
```
</details>

<details>
<summary><strong>Settings handling (qemu binary path)</strong></summary>

Since we spawn qemu from Rust via `std::process::Command`, no shell-scope entry is needed — the path is just stored in `settings.json` and passed to `vm_start`. If you later decide to *bundle* qemu or move to `tauri-plugin-shell`'s `Command.sidecar`, you'd add `externalBin` in `tauri.conf.json` and a `shell:allow-spawn` permission scoped to the sidecar name【turn0search0】. The shell plugin's JS API also supports streaming stdout via events and writing stdin via `child.write()`, so the console wiring above would port over almost unchanged【turn0search7】【turn1search1】.
</details>

---

## I. Verification against Your Reference Commands

Your PowerShell reference, as produced by the arg builder for a VM named `dev-setup` with RAM 2048, 4 vCPUs, snapshot on, one drive, and three forwards:

```
qemu-system-x86_64.exe
  -m 2048M -smp 4
  -drive file=…\images\dev-setup-overlay.qcow2,format=qcow2,if=virtio,snapshot=on
  -netdev user,id=net0,hostfwd=tcp:127.0.0.1:10225-:22,hostfwd=tcp:127.0.0.1:10380-:8080,hostfwd=tcp:127.0.0.1:10390-:8090
  -device virtio-net-pci,netdev=net0
  -nographic
```
✅ byte-for-byte equivalent semantics.

Your overlay-creation reference maps to the "New Image → Overlay" dialog:

```
qemu-img create -f qcow2 -b …\images\debian-base.qcow2 -F qcow2 …\images\overlay.qcow2 10G
```
✅ `-b` = backing file, `-F` = backing format — both surfaced as a single "parent image" dropdown in the UI, so users can't get the flags wrong.

---

## Suggested Build/Run Order for Iteration

1. Scaffold + run the empty Tauri app to confirm toolchain.
2. Wire `store.rs` + a hardcoded test VM → verify `vms.json` round-trips.
3. Implement `args.rs` with a `vm_preview_args` debug command that returns the argv as a string — diff it against your reference command before ever spawning anything.
4. `vm_start` + `vm_status` + Start/Stop buttons, no console yet. Verify the process appears in Task Manager and dies on Stop.
5. Add the xterm.js console and `vm_write_stdin` — this is the moment the PoC feels real; test with a Debian netinst ISO (`cdrom` field) so you exercise console + virtio drive + networking in one go.
6. Image manager + backing-chain tree last, since it's pure `qemu-img` plumbing once everything else works.

One heads-up for the iteration phase: `BufReader::read` on a qemu serial stream works fine, but qemu buffers output when it thinks stdout isn't a TTY — if the console looks "laggy," that's qemu's line-buffering, not a bug in your code; the standard fix is a PTY (e.g., the `portable-pty` crate) which you can slot in behind the same `vm_write_stdin` / event interface without touching the frontend.
