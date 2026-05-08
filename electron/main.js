/**
 * Electron main process — wraps the Next.js web app in a desktop shell.
 *
 * On launch:
 *   1. Spawns the FastAPI backend (uvicorn) as a child process
 *   2. Waits for the API to be ready
 *   3. Opens a BrowserWindow pointing at the Next.js dev server (dev)
 *      or the built .next/server/app (prod)
 *
 * Run locally:
 *   npm run electron:dev
 *
 * Package:
 *   npm run electron:build
 */

const { app, BrowserWindow, shell, ipcMain } = require("electron");
const path  = require("path");
const http  = require("http");
const { spawn, execSync } = require("child_process");

const IS_DEV  = !app.isPackaged;
const WEB_URL = IS_DEV ? "http://localhost:3000" : `file://${path.join(__dirname, "../.next/server/app/page.html")}`;
const API_URL = "http://localhost:8000";
const API_WAIT_TIMEOUT = 30_000;   // ms

let mainWindow = null;
let apiProcess = null;

// ── Start FastAPI backend ────────────────────────────────────────────────────

function startAPIServer() {
  const projectRoot = path.join(__dirname, "..");
  const pythonCmd   = process.platform === "win32" ? "python" : "python3";

  // Try uv first (faster), fall back to python3
  let cmd, args;
  try {
    execSync("uv --version", { stdio: "ignore" });
    cmd  = "uv";
    args = ["run", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"];
  } catch {
    cmd  = pythonCmd;
    args = ["-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"];
  }

  console.log(`[electron] Starting API: ${cmd} ${args.join(" ")}`);

  apiProcess = spawn(cmd, args, {
    cwd:   projectRoot,
    stdio: ["ignore", "pipe", "pipe"],
    env:   { ...process.env },
  });

  apiProcess.stdout.on("data", d => process.stdout.write(`[api] ${d}`));
  apiProcess.stderr.on("data", d => process.stderr.write(`[api] ${d}`));

  apiProcess.on("exit", code => {
    if (code !== 0) console.error(`[electron] API exited with code ${code}`);
  });
}

// ── Wait for API health endpoint ─────────────────────────────────────────────

function waitForAPI(timeout = API_WAIT_TIMEOUT) {
  return new Promise((resolve, reject) => {
    const start = Date.now();

    function check() {
      http.get(`${API_URL}/health`, res => {
        if (res.statusCode === 200) {
          console.log("[electron] API ready ✓");
          resolve();
        } else {
          retry();
        }
      }).on("error", () => {
        retry();
      });
    }

    function retry() {
      if (Date.now() - start > timeout) {
        reject(new Error("API did not start in time"));
        return;
      }
      setTimeout(check, 800);
    }

    check();
  });
}

// ── Browser window ───────────────────────────────────────────────────────────

function createWindow() {
  mainWindow = new BrowserWindow({
    width:          1400,
    height:         900,
    minWidth:       900,
    minHeight:      600,
    backgroundColor: "#0d0d1a",
    titleBarStyle:  "hiddenInset",
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, "preload.js"),
    },
    icon: path.join(__dirname, "../web/public/icon.png"),
  });

  // Open external links in browser, not electron
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });

  mainWindow.loadURL(WEB_URL);

  if (IS_DEV) {
    mainWindow.webContents.openDevTools({ mode: "detach" });
  }

  mainWindow.on("closed", () => { mainWindow = null; });
}

// ── App lifecycle ─────────────────────────────────────────────────────────────

app.whenReady().then(async () => {
  startAPIServer();

  try {
    await waitForAPI();
  } catch (err) {
    console.error("[electron]", err.message, "— opening window anyway");
  }

  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("quit", () => {
  if (apiProcess) {
    console.log("[electron] Stopping API server…");
    apiProcess.kill("SIGTERM");
  }
});

// ── IPC handlers (called from renderer via preload) ──────────────────────────

ipcMain.handle("get-api-url", () => API_URL);
ipcMain.handle("open-external", (_, url) => shell.openExternal(url));
