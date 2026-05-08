"use client";

import { useState, useEffect } from "react";

interface Player {
  element_id: string;
  name: string;
  position: string;
  team_name: string;
  price: string;
  xp: string;
  p_start: string;
  p_sub: string;
  ownership_pct: string;
  status: string;
}

const POS_COLORS: Record<string, string> = {
  GKP: "bg-yellow-100 text-yellow-800",
  DEF: "bg-green-100 text-green-800",
  MID: "bg-blue-100 text-blue-800",
  FWD: "bg-red-100 text-red-800",
};

const STATUS_DOT: Record<string, string> = {
  a: "bg-green-400",
  d: "bg-yellow-400",
  i: "bg-red-400",
  n: "bg-gray-400",
};

function XpBar({ value, max = 10 }: { value: number; max?: number }) {
  const pct = Math.min((value / max) * 100, 100);
  const color =
    value >= 7 ? "bg-green-500" : value >= 5 ? "bg-blue-500" : value >= 3 ? "bg-yellow-500" : "bg-gray-300";
  return (
    <div className="flex items-center gap-2">
      <div className="h-2 w-20 rounded bg-gray-200">
        <div className={`h-2 rounded ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-sm font-mono font-semibold">{value.toFixed(2)}</span>
    </div>
  );
}

export default function PlayerTable() {
  const [players, setPlayers] = useState<Player[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [position, setPosition] = useState("all");
  const [search, setSearch] = useState("");

  useEffect(() => {
    setLoading(true);
    fetch(`/api/predictions?position=${position}&limit=100`)
      .then((r) => r.json())
      .then((data) => {
        if (data.error) throw new Error(data.error);
        setPlayers(data.predictions ?? []);
        setError(null);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [position]);

  const filtered = players.filter((p) =>
    p.name?.toLowerCase().includes(search.toLowerCase()) ||
    p.team_name?.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="space-y-4">
      {/* Filters */}
      <div className="flex flex-wrap gap-3 items-center">
        <input
          type="text"
          placeholder="Search player or team..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="border border-gray-200 rounded-lg px-3 py-2 text-sm w-64 focus:outline-none focus:ring-2 focus:ring-indigo-400"
        />
        <div className="flex gap-1">
          {["all", "GKP", "DEF", "MID", "FWD"].map((pos) => (
            <button
              key={pos}
              onClick={() => setPosition(pos)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
                position === pos
                  ? "bg-indigo-600 text-white"
                  : "bg-gray-100 text-gray-600 hover:bg-gray-200"
              }`}
            >
              {pos}
            </button>
          ))}
        </div>
        <span className="text-xs text-gray-400">{filtered.length} players</span>
      </div>

      {/* Error */}
      {error && (
        <div className="rounded-lg bg-red-50 border border-red-200 p-4 text-sm text-red-700">
          {error}
        </div>
      )}

      {/* Table */}
      {loading ? (
        <div className="flex items-center justify-center h-40 text-gray-400 text-sm">Loading predictions...</div>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-gray-200">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-50 text-gray-500 text-xs uppercase tracking-wide">
                <th className="px-4 py-3 text-left">#</th>
                <th className="px-4 py-3 text-left">Player</th>
                <th className="px-4 py-3 text-left">Pos</th>
                <th className="px-4 py-3 text-left">Team</th>
                <th className="px-4 py-3 text-right">Price</th>
                <th className="px-4 py-3 text-left">xP</th>
                <th className="px-4 py-3 text-right">P(Start)</th>
                <th className="px-4 py-3 text-right">Own%</th>
                <th className="px-4 py-3 text-center">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {filtered.slice(0, 50).map((p, i) => {
                const xpVal = parseFloat(p.xp ?? "0");
                const pStart = parseFloat(p.p_start ?? "0");
                const own = parseFloat(p.ownership_pct ?? "0");
                const price = (parseInt(p.price ?? "0") / 10).toFixed(1);
                const status = p.status ?? "a";
                return (
                  <tr key={p.element_id} className="hover:bg-gray-50 transition-colors">
                    <td className="px-4 py-3 text-gray-400 font-mono text-xs">{i + 1}</td>
                    <td className="px-4 py-3 font-semibold text-gray-800">{p.name}</td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-0.5 rounded text-xs font-bold ${POS_COLORS[p.position] ?? ""}`}>
                        {p.position}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-gray-500">{p.team_name}</td>
                    <td className="px-4 py-3 text-right font-mono text-gray-700">£{price}m</td>
                    <td className="px-4 py-3">
                      <XpBar value={xpVal} />
                    </td>
                    <td className="px-4 py-3 text-right text-gray-600">{(pStart * 100).toFixed(0)}%</td>
                    <td className="px-4 py-3 text-right text-gray-600">{own.toFixed(1)}%</td>
                    <td className="px-4 py-3 text-center">
                      <span className={`inline-block w-2.5 h-2.5 rounded-full ${STATUS_DOT[status] ?? "bg-gray-300"}`} title={status} />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
