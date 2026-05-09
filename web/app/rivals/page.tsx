"use client";
import { useState } from "react";
import { api } from "@/lib/api";

export default function RivalsPage() {
  const [leagueId, setLeagueId] = useState("");
  const [managerId, setManagerId] = useState("");
  const [report, setReport] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  async function handleAnalyse() {
    if (!leagueId || !managerId) return;
    setLoading(true);
    try {
      const result = await api.rivals({
        league_id: +leagueId,
        manager_id: +managerId,
        gameweek: 30, // TODO: auto-detect
      });
      setReport(result);
    } catch (e: any) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-5xl">
      <h1 className="text-2xl font-bold text-fpl-green mb-1">Rival Intelligence</h1>
      <p className="text-sm text-gray-400 mb-6">
        Track mini-league rivals, find differentials, counter-optimize your squad
      </p>

      <div className="bg-fpl-card border border-fpl-border rounded-xl p-6 mb-6">
        <div className="flex gap-4 flex-wrap items-end">
          <div>
            <label className="text-xs text-gray-400 uppercase tracking-wide font-semibold block mb-1">League ID</label>
            <input type="number" value={leagueId} onChange={(e) => setLeagueId(e.target.value)}
              placeholder="e.g. 314"
              className="bg-fpl-bg border border-fpl-border rounded-lg px-3 py-2 text-sm text-white w-36 focus:outline-none focus:ring-2 focus:ring-fpl-green/50" />
          </div>
          <div>
            <label className="text-xs text-gray-400 uppercase tracking-wide font-semibold block mb-1">My Manager ID</label>
            <input type="number" value={managerId} onChange={(e) => setManagerId(e.target.value)}
              placeholder="e.g. 742663"
              className="bg-fpl-bg border border-fpl-border rounded-lg px-3 py-2 text-sm text-white w-40 focus:outline-none focus:ring-2 focus:ring-fpl-green/50" />
          </div>
          <button onClick={handleAnalyse} disabled={loading}
            className="bg-fpl-green text-fpl-purple font-bold px-6 py-2 rounded-lg text-sm hover:bg-fpl-green/90 transition-colors disabled:opacity-50">
            {loading ? "Analysing..." : "Analyse Rivals"}
          </button>
        </div>
      </div>

      {report && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
            {[
              { label: "My Points", value: report.my_points, color: "text-white" },
              { label: "Gap to Leader", value: report.gap_to_leader, color: report.gap_to_leader > 0 ? "text-red-400" : "text-green-400" },
              { label: "Strategy", value: report.strategy, color: "text-fpl-cyan" },
              { label: "GWs Left", value: report.gws_remaining, color: "text-gray-300" },
            ].map(s => (
              <div key={s.label} className="bg-fpl-card border border-fpl-border rounded-xl p-4">
                <div className="text-xs text-gray-500 uppercase tracking-wide font-semibold mb-1">{s.label}</div>
                <div className={`text-2xl font-bold ${s.color}`}>{s.value}</div>
              </div>
            ))}
          </div>

          {report.rivals?.length > 0 && (
            <div className="bg-fpl-card border border-fpl-border rounded-xl overflow-hidden">
              <h3 className="text-sm font-semibold text-gray-300 px-4 py-3 border-b border-fpl-border">Top Rivals</h3>
              <table className="w-full text-sm">
                <thead className="bg-fpl-bg text-gray-400 text-xs uppercase">
                  <tr>
                    <th className="px-4 py-2 text-left">Rank</th>
                    <th className="px-4 py-2 text-left">Name</th>
                    <th className="px-4 py-2 text-right">Points</th>
                  </tr>
                </thead>
                <tbody>
                  {report.rivals.map((r: any, i: number) => (
                    <tr key={i} className="border-t border-fpl-border">
                      <td className="px-4 py-2 text-gray-400">{r.rank}</td>
                      <td className="px-4 py-2">{r.name}</td>
                      <td className="px-4 py-2 text-right font-mono">{r.total_points}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {!report && !loading && (
        <div className="border border-dashed border-fpl-border rounded-xl p-12 text-center text-gray-500 text-sm">
          Enter your league and manager IDs to analyse your rivals.
        </div>
      )}
    </div>
  );
}
