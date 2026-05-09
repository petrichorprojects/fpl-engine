"use client";
import { useState } from "react";

const SIGNAL_COLORS: Record<string, string> = {
  INJURY_OUT: "bg-red-500/20 text-red-400",
  INJURY_DOUBT: "bg-orange-500/20 text-orange-400",
  ROTATION_RISK: "bg-yellow-500/20 text-yellow-400",
  ROTATION_LIKELY: "bg-yellow-500/20 text-yellow-300",
  CONFIRMED_FIT: "bg-green-500/20 text-green-400",
  RETURNING: "bg-blue-500/20 text-blue-400",
};

export default function PressersPage() {
  const [manager, setManager] = useState("");
  const [team, setTeam] = useState("");
  const [text, setText] = useState("");
  const [signals, setSignals] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  async function handleAnalyse() {
    if (!text.trim()) return;
    setLoading(true);
    try {
      const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
      // Direct call to Python API since this endpoint may not be in the Next.js routes
      const res = await fetch(`${API_URL}/pressers/add`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ manager, team, text, gameweek: 30 }),
      });
      if (res.ok) {
        const data = await res.json();
        setSignals(data.signals || []);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-4xl">
      <h1 className="text-2xl font-bold text-fpl-green mb-1">Press Conference Analyser</h1>
      <p className="text-sm text-gray-400 mb-6">
        Paste manager press conference text to extract injury/rotation signals
      </p>

      <div className="bg-fpl-card border border-fpl-border rounded-xl p-6 mb-6">
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <label className="text-xs text-gray-400 uppercase tracking-wide font-semibold block mb-1">Manager</label>
            <input value={manager} onChange={(e) => setManager(e.target.value)}
              placeholder="e.g. Arteta"
              className="w-full bg-fpl-bg border border-fpl-border rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-fpl-green/50" />
          </div>
          <div>
            <label className="text-xs text-gray-400 uppercase tracking-wide font-semibold block mb-1">Team</label>
            <input value={team} onChange={(e) => setTeam(e.target.value)}
              placeholder="e.g. Arsenal"
              className="w-full bg-fpl-bg border border-fpl-border rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-fpl-green/50" />
          </div>
        </div>
        <div className="mb-4">
          <label className="text-xs text-gray-400 uppercase tracking-wide font-semibold block mb-1">Transcript</label>
          <textarea value={text} onChange={(e) => setText(e.target.value)} rows={6}
            placeholder="Paste press conference transcript here..."
            className="w-full bg-fpl-bg border border-fpl-border rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-fpl-green/50 resize-y" />
        </div>
        <button onClick={handleAnalyse} disabled={loading || !text.trim()}
          className="bg-fpl-green text-fpl-purple font-bold px-6 py-2 rounded-lg text-sm hover:bg-fpl-green/90 transition-colors disabled:opacity-50">
          {loading ? "Analysing..." : "Extract Signals"}
        </button>
      </div>

      {signals.length > 0 && (
        <div className="bg-fpl-card border border-fpl-border rounded-xl overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-fpl-bg text-gray-400 text-xs uppercase">
              <tr>
                <th className="px-4 py-3 text-left">Player</th>
                <th className="px-4 py-3 text-left">Signal</th>
                <th className="px-4 py-3 text-right">Confidence</th>
                <th className="px-4 py-3 text-left">Quote</th>
              </tr>
            </thead>
            <tbody>
              {signals.map((s, i) => (
                <tr key={i} className="border-t border-fpl-border">
                  <td className="px-4 py-3 font-medium">{s.player_name}</td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-0.5 rounded text-xs font-bold ${SIGNAL_COLORS[s.signal_type] || 'text-gray-400'}`}>
                      {s.signal_type}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right font-mono">{(s.confidence * 100).toFixed(0)}%</td>
                  <td className="px-4 py-3 text-gray-500 text-xs max-w-xs truncate">{s.raw_quote}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {signals.length === 0 && !loading && (
        <div className="border border-dashed border-fpl-border rounded-xl p-12 text-center text-gray-500 text-sm">
          Paste a press conference transcript above and click "Extract Signals" to find injury and rotation information.
        </div>
      )}
    </div>
  );
}
