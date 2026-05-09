"use client";
import PlayerTable from "@/components/PlayerTable";

export default function PlayersPage() {
  return (
    <div className="max-w-6xl">
      <h1 className="text-2xl font-bold text-fpl-green mb-1">Player Predictions</h1>
      <p className="text-sm text-gray-400 mb-6">
        Combined xP = P(start) × E[pts|start] + P(sub) × E[pts|sub]
      </p>
      <PlayerTable />
    </div>
  );
}
