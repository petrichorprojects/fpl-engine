"use client";
import { useState, useEffect } from "react";
import { api } from "@/lib/api";

export default function SettingsPage() {
  const [health, setHealth] = useState<any>(null);
  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

  useEffect(() => {
    api.health().then(setHealth).catch(console.error);
  }, []);

  return (
    <div className="max-w-3xl">
      <h1 className="text-2xl font-bold text-fpl-green mb-1">Settings</h1>
      <p className="text-sm text-gray-400 mb-6">Engine status and configuration</p>

      <div className="space-y-4">
        <div className="bg-fpl-card border border-fpl-border rounded-xl p-5">
          <h3 className="text-sm font-semibold text-gray-300 mb-3 uppercase tracking-wide">Engine Status</h3>
          {health ? (
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div className="text-gray-400">Status</div>
              <div className="text-green-400 font-semibold">{health.status}</div>
              <div className="text-gray-400">Models Trained</div>
              <div className={health.models_trained ? "text-green-400" : "text-yellow-400"}>
                {health.models_trained ? "Yes ✓" : "Not yet"}
              </div>
              <div className="text-gray-400">Predictions Available</div>
              <div className={health.predictions_available ? "text-green-400" : "text-yellow-400"}>
                {health.predictions_available ? "Yes ✓" : "Not yet"}
              </div>
              <div className="text-gray-400">Current Gameweek</div>
              <div>{health.current_gw || "—"}</div>
              <div className="text-gray-400">Players Loaded</div>
              <div>{health.n_players || "—"}</div>
            </div>
          ) : (
            <div className="text-gray-500 text-sm animate-pulse">Checking engine status...</div>
          )}
        </div>

        <div className="bg-fpl-card border border-fpl-border rounded-xl p-5">
          <h3 className="text-sm font-semibold text-gray-300 mb-3 uppercase tracking-wide">API Connection</h3>
          <div className="text-sm">
            <div className="text-gray-400 mb-1">Backend URL</div>
            <code className="text-fpl-cyan text-xs bg-fpl-bg px-2 py-1 rounded">{apiUrl}</code>
          </div>
          <div className="text-sm mt-3">
            <div className="text-gray-400 mb-1">Swagger Docs</div>
            <a href={`${apiUrl}/docs`} target="_blank" rel="noopener"
              className="text-fpl-green text-xs hover:underline">{apiUrl}/docs →</a>
          </div>
        </div>

        <div className="bg-fpl-card border border-fpl-border rounded-xl p-5">
          <h3 className="text-sm font-semibold text-gray-300 mb-3 uppercase tracking-wide">About</h3>
          <div className="text-sm text-gray-400 space-y-1">
            <p>FPL Engine v0.1.0</p>
            <p>Minutes Model (XGBoost) + Points Model (XGBoost) + MILP Optimizer (PuLP/HiGHS)</p>
            <p>
              <a href="https://github.com/petrichorprojects/fpl-engine" target="_blank" rel="noopener"
                className="text-fpl-green hover:underline">GitHub →</a>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
