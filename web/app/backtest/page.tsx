"use client";
import { useState } from "react";
import { api, BacktestResult } from "@/lib/api";
import PointsChart from "@/components/PointsChart";

const STRATEGIES = [
  { id: "engine", label: "Full Engine", desc: "Minutes + Points + MILP optimizer" },
  { id: "top_form", label: "Top Form", desc: "Pick highest FPL form metric" },
  { id: "high_ownership", label: "Template", desc: "Most-owned players (crowd wisdom)" },
  { id: "random", label: "Random", desc: "Random valid squad (baseline)" },
];

export default function BacktestPage() {
  const [strategy, setStrategy] = useState("top_form");
  const [startGw, setStartGw] = useState(5);
  const [endGw, setEndGw] = useState(38);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleRun() {
    setLoading(true);
    try {
      const r = await api.backtest({ strategy, start_gw: startGw, end_gw: endGw });
      setResult(r);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-5xl">
      <h1 className="text-2xl font-bold text-fpl-green mb-1">Season Backtest</h1>
      <p className="text-sm text-gray-400 mb-6">
        Walk-forward replay: train on GW 1→N-1, predict GW N, compare strategies
      </p>

      <div className="bg-fpl-card border border-fpl-border rounded-xl p-6 mb-6">
        <div className="grid grid-cols-1 sm:grid-cols-4 gap-4 mb-4">
          <div className="sm:col-span-2">
            <label className="text-xs text-gray-400 uppercase tracking-wide font-semibold block mb-1">Strategy</label>
            <select value={strategy} onChange={(e) => setStrategy(e.target.value)}
              className="w-full bg-fpl-bg border border-fpl-border rounded-lg px-3 py-2 text-sm text-white">
              {STRATEGIES.map(s => <option key={s.id} value={s.id}>{s.label} — {s.desc}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs text-gray-400 uppercase tracking-wide font-semibold block mb-1">Start GW</label>
            <input type="number" value={startGw} onChange={(e) => setStartGw(+e.target.value)} min={1} max={38}
              className="w-full bg-fpl-bg border border-fpl-border rounded-lg px-3 py-2 text-sm text-white" />
          </div>
          <div>
            <label className="text-xs text-gray-400 uppercase tracking-wide font-semibold block mb-1">End GW</label>
            <input type="number" value={endGw} onChange={(e) => setEndGw(+e.target.value)} min={1} max={38}
              className="w-full bg-fpl-bg border border-fpl-border rounded-lg px-3 py-2 text-sm text-white" />
          </div>
        </div>
        <button onClick={handleRun} disabled={loading}
          className="bg-fpl-green text-fpl-purple font-bold px-6 py-2 rounded-lg text-sm hover:bg-fpl-green/90 transition-colors disabled:opacity-50">
          {loading ? "Running backtest..." : "Run Backtest"}
        </button>
      </div>

      {result && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="bg-fpl-card border border-fpl-border rounded-xl p-4">
              <div className="text-xs text-gray-500 uppercase font-semibold mb-1">Total Points</div>
              <div className="text-3xl font-bold text-fpl-green">{result.total_points}</div>
            </div>
            <div className="bg-fpl-card border border-fpl-border rounded-xl p-4">
              <div className="text-xs text-gray-500 uppercase font-semibold mb-1">Avg / GW</div>
              <div className="text-3xl font-bold text-white">{result.avg_gw_points}</div>
            </div>
            <div className="bg-fpl-card border border-fpl-border rounded-xl p-4">
              <div className="text-xs text-gray-500 uppercase font-semibold mb-1">Spearman ρ</div>
              <div className="text-3xl font-bold text-fpl-cyan">{result.avg_spearman ?? "N/A"}</div>
            </div>
          </div>

          {result.gw_results?.length > 0 && (
            <div className="bg-fpl-card border border-fpl-border rounded-xl p-5">
              <h3 className="text-sm font-semibold text-gray-300 mb-4">Cumulative Points</h3>
              <PointsChart data={result.gw_results.map(g => ({
                gameweek: g.gameweek,
                points: g.cumulative_points,
              }))} />
            </div>
          )}
        </div>
      )}

      {!result && !loading && (
        <div className="border border-dashed border-fpl-border rounded-xl p-12 text-center text-gray-500 text-sm">
          Select a strategy and click "Run Backtest" to simulate a season.
        </div>
      )}
    </div>
  );
}
