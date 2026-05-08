"use client";

import { useState } from "react";
import PlayerTable from "@/components/PlayerTable";
import SquadBuilder from "@/components/SquadBuilder";

type Tab = "predictions" | "squad" | "rivals" | "calendar";

const TABS: { id: Tab; label: string; icon: string }[] = [
  { id: "predictions", label: "xP Predictions",   icon: "📊" },
  { id: "squad",       label: "Squad Optimiser",   icon: "⚡" },
  { id: "rivals",      label: "Rival Tracker",     icon: "🎯" },
  { id: "calendar",    label: "Fixture Calendar",  icon: "📅" },
];

function RivalTracker() {
  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-gray-200 p-6 bg-white">
        <h3 className="font-semibold text-gray-800 mb-2">Mini-League Intelligence</h3>
        <p className="text-sm text-gray-500 mb-4">
          Enter your league ID to track rivals, identify differentials, and get counter-optimised picks.
        </p>
        <div className="flex gap-3 flex-wrap">
          <input type="number" placeholder="League ID"
            className="border border-gray-200 rounded-lg px-3 py-2 text-sm w-40 focus:outline-none focus:ring-2 focus:ring-indigo-400" />
          <input type="number" placeholder="My Manager ID"
            className="border border-gray-200 rounded-lg px-3 py-2 text-sm w-44 focus:outline-none focus:ring-2 focus:ring-indigo-400" />
          <button className="bg-indigo-600 hover:bg-indigo-700 text-white font-semibold px-4 py-2 rounded-lg text-sm transition-colors">
            Analyse Rivals
          </button>
        </div>
      </div>
      <div className="rounded-xl border border-dashed border-gray-200 p-8 text-center text-gray-400 text-sm">
        Rival analysis will appear here after running the engine with a league ID.
      </div>
    </div>
  );
}

function FixtureCalendar() {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[
          { label: "Next DGW", value: "GW34", note: "7 teams have 2 fixtures", color: "text-green-600" },
          { label: "Next BGW", value: "GW27", note: "5 teams blank (FA Cup)", color: "text-red-600" },
          { label: "Chip Rec.", value: "Bench Boost → GW34", note: "Score: 4.2 / 5.0", color: "text-indigo-600" },
        ].map((card) => (
          <div key={card.label} className="rounded-xl border border-gray-200 bg-white p-5">
            <div className="text-xs text-gray-500 uppercase tracking-wide font-semibold mb-1">{card.label}</div>
            <div className={`text-2xl font-bold ${card.color}`}>{card.value}</div>
            <div className="text-xs text-gray-400 mt-1">{card.note}</div>
          </div>
        ))}
      </div>
      <div className="rounded-xl border border-dashed border-gray-200 p-8 text-center text-gray-400 text-sm">
        Full fixture calendar loads after engine data fetch.
      </div>
    </div>
  );
}

export default function Home() {
  const [tab, setTab] = useState<Tab>("predictions");

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-sm">FE</span>
              </div>
              <div>
                <span className="font-bold text-gray-900 text-lg">FPL Engine</span>
                <span className="ml-2 text-xs text-gray-400 font-medium hidden sm:inline">
                  Minutes Model · Points Model · Meta Optimizer
                </span>
              </div>
            </div>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-green-100 text-green-700 text-xs font-medium">
              <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
              Live
            </span>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
          {[
            { label: "Players Modelled", value: "800+",    icon: "👤" },
            { label: "Features",         value: "100+",    icon: "📐" },
            { label: "Model Type",       value: "XGBoost", icon: "🤖" },
            { label: "Optimizer",        value: "MILP",    icon: "⚙️" },
          ].map((stat) => (
            <div key={stat.label} className="rounded-xl border border-gray-200 bg-white p-4">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-lg">{stat.icon}</span>
                <span className="text-xs text-gray-500 font-medium uppercase tracking-wide">{stat.label}</span>
              </div>
              <div className="text-xl font-bold text-gray-900">{stat.value}</div>
            </div>
          ))}
        </div>

        <div className="flex gap-1 mb-6 bg-gray-100 rounded-xl p-1 w-fit">
          {TABS.map((t) => (
            <button key={t.id} onClick={() => setTab(t.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                tab === t.id ? "bg-white text-gray-900 shadow-sm" : "text-gray-500 hover:text-gray-700"
              }`}>
              <span>{t.icon}</span>
              {t.label}
            </button>
          ))}
        </div>

        <div className="bg-white rounded-2xl border border-gray-200 p-6">
          {tab === "predictions" && (
            <div>
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-lg font-bold text-gray-900">Expected Points — Next Gameweek</h2>
                  <p className="text-sm text-gray-500 mt-0.5">
                    Combined: P(start) × E[pts|start] + P(sub) × E[pts|sub]
                  </p>
                </div>
                <a href="/api/predictions?limit=500" target="_blank"
                  className="text-xs text-indigo-600 hover:underline font-medium">Export JSON →</a>
              </div>
              <PlayerTable />
            </div>
          )}
          {tab === "squad" && (
            <div>
              <div className="mb-6">
                <h2 className="text-lg font-bold text-gray-900">Ownership-Aware Squad Optimiser</h2>
                <p className="text-sm text-gray-500 mt-0.5">
                  MILP optimizer with gamestate-adaptive differential weighting
                </p>
              </div>
              <SquadBuilder />
            </div>
          )}
          {tab === "rivals" && (
            <div>
              <div className="mb-6">
                <h2 className="text-lg font-bold text-gray-900">Rival Intelligence</h2>
                <p className="text-sm text-gray-500 mt-0.5">Track rivals and find counter-optimization opportunities</p>
              </div>
              <RivalTracker />
            </div>
          )}
          {tab === "calendar" && (
            <div>
              <div className="mb-6">
                <h2 className="text-lg font-bold text-gray-900">Fixture Calendar & Chip Planner</h2>
                <p className="text-sm text-gray-500 mt-0.5">DGW/BGW detection and chip timing optimization</p>
              </div>
              <FixtureCalendar />
            </div>
          )}
        </div>

        <div className="mt-6 text-center text-xs text-gray-400">
          FPL Engine · Minutes Model + Points Model + Meta Optimizer · Data: FPL API + Understat
        </div>
      </div>
    </div>
  );
}
