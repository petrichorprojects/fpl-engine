/**
 * Electron preload — exposes a minimal safe bridge to the renderer.
 */
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("electron", {
  getApiUrl:    () => ipcRenderer.invoke("get-api-url"),
  openExternal: (url) => ipcRenderer.invoke("open-external", url),
  isElectron:   true,
});
