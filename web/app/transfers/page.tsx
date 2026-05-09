"use client";
import { useState } from "react";
import { api } from "@/lib/api";

export default function TransfersPage() {
  const [teamId, setTeamId] = useState("");
  const [freeTransfers, setFreeTransfers] = useState(1);
  const [bank, setBank] = useState(0);
  const [transfers, setTransfers] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  async function handleOptimize() {
    if (!teamId) return;
    setLoading(true);
    try {
      // For now, use placeholder squad IDs — in production, fetch from FPL API
      const result = await api.optimizeTransfers({
        squad_ids: [], // Will be populated from team ID
        free_transfers: freeTransfers,
        bank: bank,
        horizon: 3,
      });
      setTransfers(result);
    } catch (e: any) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-4xl">
      <h1 className="text-2xl font-bold text-fpl-green mb-1">Transfer Planner</h1>
      <p className="text-sm text-gray-400 mb-6">
        Multi-week transfer optimization with hit penalty accounting
      </p>

      <div className="bg-fpl-card border border-fpl-border rounded-xl p-6 mb-6">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-4">
          <div>
            <label className="text-xs text-gray-400 uppercase tracking-wide font-semibold block mb-1">FPL Team ID</label>
            <input type="number" value={teamId} onChange={(e) => setTeamId(e.target.value)}
              placeholder="e.g. 742663"
              className="w-full bg-fpl-bg border border-fpl-border rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-fpl-green/50" />
          </div>
          <div>
            <label className="text-xs text-gray-400 uppercase tracking-wide font-semibold block mb-1">Free Transfers</label>
            <select value={freeTransfers} onChange={(e) => setFreeTransfers(+e.target.value)}
              className="w-full bg-fpl-bg border border-fpl-border rounded-lg px-3 py-2 text-sm text-white">
              {[1,2,3,4,5].map(n => <option key={n} value={n}>{n}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs text-gray-400 uppercase tracking-wide font-semibold block mb-1">Bank (£m)</label>
            <input type="number" step="0.1" value={bank/10} onChange={(e) => setBank(Math.round(+e.target.value * 10))}
              className="w-full bg-fpl-bg border border-fpl-border rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-fpl-green/50" />
          </div>
        </div>
        <button onClick={handleOptimize} disabled={loading}
          className="bg-fpl-green text-fpl-purple font-bold px-6 py-2 rounded-lg text-sm hover:bg-fpl-green/90 transition-colors disabled:opacity-50">
          {loading ? "Optimizing..." : "Find Optimal Transfers"}
        </button>
      </div>

      {transfers.length > 0 && (
        <div className="bg-fpl-card border border-fpl-border rounded-xl overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-fpl-bg text-gray-400 text-xs uppercase">
              <tr>
                <th className="px-4 py-3 text-left">Out</th>
                <th className="px-4 py-3 text-left">In</th>
                <th className="px-4 py-3 text-right">xP Gain</th>
                <th className="px-4 py-3 text-right">Type</th>
              </tr>
            </thead>
            <tbody>
              {transfers.map((t, i) => (
                <tr key={i} className="border-t border-fpl-border">
                  <td className="px-4 py-3 text-red-400">{t.out_name}</td>
                  <td className="px-4 py-3 text-green-400">{t.in_name}</td>
                  <td className="px-4 py-3 text-right font-mono">{t.xp_gain > 0 ? '+' : ''}{t.xp_gain.toFixed(1)}</td>
                  <td className="px-4 py-3 text-right">
                    <span className={`px-2 py-0.5 rounded text-xs font-semibold ${t.hit ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'}`}>
                      {t.hit ? "HIT -4" : "FREE"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {transfers.length === 0 && !loading && (
        <div className="border border-dashed border-fpl-border rounded-xl p-12 text-center text-gray-500 text-sm">
          Enter your FPL Team ID and click optimize to get transfer suggestions.
        </div>
      )}
    </div>
  );
}
