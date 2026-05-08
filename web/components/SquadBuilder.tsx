"use client";

import { useState } from "react";

type GameState = "neutral" | "leading" | "chasing" | "mini_league";
type Chip = "none" | "wildcard" | "free_hit" | "bench_boost" | "triple_captain";

interface EngineConfig {
  gamestate: GameState;
  budget: number;
  chip: Chip;
}

const GAMESTATE_INFO: Record<GameState, { label: string; desc: string; color: string }> = {
  neutral:     { label: "Neutral",     desc: "Balanced approach",                color: "border-gray-300 bg-gray-50" },
  leading:     { label: "Leading",     desc: "Match template, protect your rank", color: "border-green-300 bg-green-50" },
  chasing:     { label: "Chasing",     desc: "Go differential, accept variance",  color: "border-orange-300 bg-orange-50" },
  mini_league: { label: "Mini-League", desc: "Target rival weaknesses",           color: "border-purple-300 bg-purple-50" },
};

const CHIP_INFO: Record<Chip, { label: string; icon: string }> = {
  none:           { label: "No Chip",       icon: "—" },
  wildcard:       { label: "Wildcard",      icon: "🃏" },
  free_hit:       { label: "Free Hit",      icon: "⚡" },
  bench_boost:    { label: "Bench Boost",   icon: "📈" },
  triple_captain: { label: "Triple Captain",icon: "👑" },
};

export default function SquadBuilder() {
  const [config, setConfig] = useState<EngineConfig>({
    gamestate: "neutral",
    budget: 1000,
    chip: "none",
  });
  const [running, setRunning] = useState(false);
  const [output, setOutput] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setRunning(true);
    setOutput(null);
    setError(null);
    try {
      const res = await fetch("/api/engine", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      const data = await res.json();
      if (data.success) {
        setOutput(data.output);
      } else {
        setError(data.error ?? "Engine failed");
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Network error");
    } finally {
      setRunning(false);
    }
  };

  const budgetM = (config.budget / 10).toFixed(1);

  return (
    <div className="space-y-6">
      {/* Gamestate selector */}
      <div>
        <label className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2 block">
          Strategy
        </label>
        <div className="grid grid-cols-2 gap-2">
          {(Object.entries(GAMESTATE_INFO) as [GameState, typeof GAMESTATE_INFO[GameState]][]).map(
            ([gs, info]) => (
              <button
                key={gs}
                onClick={() => setConfig((c) => ({ ...c, gamestate: gs }))}
                className={`border-2 rounded-xl p-3 text-left transition-all ${
                  config.gamestate === gs
                    ? info.color + " border-opacity-100"
                    : "border-gray-200 bg-white hover:border-gray-300"
                }`}
              >
                <div className="font-semibold text-sm text-gray-800">{info.label}</div>
                <div className="text-xs text-gray-500 mt-0.5">{info.desc}</div>
              </button>
            )
          )}
        </div>
      </div>

      {/* Budget slider */}
      <div>
        <label className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2 block">
          Budget — £{budgetM}m
        </label>
        <input
          type="range"
          min={900}
          max={1050}
          step={5}
          value={config.budget}
          onChange={(e) => setConfig((c) => ({ ...c, budget: parseInt(e.target.value) }))}
          className="w-full accent-indigo-600"
        />
        <div className="flex justify-between text-xs text-gray-400 mt-1">
          <span>£90.0m</span>
          <span>£105.0m</span>
        </div>
      </div>

      {/* Chip selector */}
      <div>
        <label className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2 block">
          Active Chip
        </label>
        <div className="flex flex-wrap gap-2">
          {(Object.entries(CHIP_INFO) as [Chip, typeof CHIP_INFO[Chip]][]).map(([chip, info]) => (
            <button
              key={chip}
              onClick={() => setConfig((c) => ({ ...c, chip }))}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium border transition-colors ${
                config.chip === chip
                  ? "bg-indigo-600 text-white border-indigo-600"
                  : "bg-white text-gray-600 border-gray-200 hover:border-indigo-300"
              }`}
            >
              {info.icon} {info.label}
            </button>
          ))}
        </div>
      </div>

      {/* Run button */}
      <button
        onClick={run}
        disabled={running}
        className="w-full bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-400 text-white font-semibold py-3 rounded-xl transition-colors flex items-center justify-center gap-2"
      >
        {running ? (
          <>
            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
            </svg>
            Running engine…
          </>
        ) : (
          "⚡ Optimise Squad"
        )}
      </button>

      {/* Error */}
      {error && (
        <div className="rounded-xl bg-red-50 border border-red-200 p-4 text-sm text-red-700">
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Output */}
      {output && (
        <div className="rounded-xl bg-gray-900 p-4 overflow-x-auto">
          <pre className="text-green-400 text-xs font-mono whitespace-pre-wrap">{output}</pre>
        </div>
      )}
    </div>
  );
}
