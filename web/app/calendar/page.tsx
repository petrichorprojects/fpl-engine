"use client";
import { useState, useEffect } from "react";
import { api, CalendarData } from "@/lib/api";

export default function CalendarPage() {
  const [data, setData] = useState<CalendarData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.calendar().then(setData).catch(console.error).finally(() => setLoading(false));
  }, []);

  return (
    <div className="max-w-5xl">
      <h1 className="text-2xl font-bold text-fpl-green mb-1">Fixture Calendar</h1>
      <p className="text-sm text-gray-400 mb-6">
        DGW/BGW detection, prediction, and chip timing optimization
      </p>

      {loading ? (
        <div className="text-gray-500 text-sm animate-pulse">Loading calendar data...</div>
      ) : data ? (
        <div className="space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="bg-fpl-card border border-fpl-border rounded-xl p-5">
              <h3 className="text-sm font-semibold text-green-400 mb-3 uppercase tracking-wide">Double Gameweeks</h3>
              {Object.keys(data.doubles).length > 0 ? (
                <div className="space-y-2">
                  {Object.entries(data.doubles).map(([gw, teams]) => (
                    <div key={gw} className="flex justify-between text-sm">
                      <span className="text-gray-300">GW{gw}</span>
                      <span className="text-gray-500">{teams.length} teams</span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-500 text-sm">No confirmed DGWs yet</p>
              )}
            </div>
            <div className="bg-fpl-card border border-fpl-border rounded-xl p-5">
              <h3 className="text-sm font-semibold text-red-400 mb-3 uppercase tracking-wide">Blank Gameweeks</h3>
              {Object.keys(data.blanks).length > 0 ? (
                <div className="space-y-2">
                  {Object.entries(data.blanks).map(([gw, teams]) => (
                    <div key={gw} className="flex justify-between text-sm">
                      <span className="text-gray-300">GW{gw}</span>
                      <span className="text-gray-500">{teams.length} teams blank</span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-500 text-sm">No confirmed BGWs</p>
              )}
            </div>
          </div>

          {data.predicted_events?.length > 0 && (
            <div className="bg-fpl-card border border-fpl-border rounded-xl p-5">
              <h3 className="text-sm font-semibold text-fpl-cyan mb-3 uppercase tracking-wide">Predicted Events</h3>
              <div className="space-y-2">
                {data.predicted_events.map((e, i) => (
                  <div key={i} className="flex items-center gap-3 text-sm">
                    <span className={`px-2 py-0.5 rounded text-xs font-bold ${e.event_type === 'DGW' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
                      {e.event_type}
                    </span>
                    <span className="text-gray-300">GW{e.gameweek}</span>
                    <span className="text-gray-500">{e.reason}</span>
                    <span className="ml-auto text-gray-400">{(e.confidence * 100).toFixed(0)}% conf</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="border border-dashed border-fpl-border rounded-xl p-12 text-center text-gray-500 text-sm">
          Calendar data will load after the engine fetches fixture data. Run the engine first.
        </div>
      )}
    </div>
  );
}
