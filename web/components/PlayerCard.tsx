"use client";

import { type Player } from "@/lib/api";
import { clsx } from "clsx";
import { Star } from "lucide-react";

const STATUS_DOT: Record<string, string> = {
  a: "bg-green-400",
  d: "bg-yellow-400",
  i: "bg-red-500",
  s: "bg-orange-500",
  u: "bg-gray-500",
};

interface Props {
  player: Player;
  showCaptainStar?: boolean;
  highlight?: "captain" | "vice" | "transfer-in" | "transfer-out";
}

export function PlayerCard({ player, showCaptainStar, highlight }: Props) {
  const dotColor = STATUS_DOT[player.status] ?? "bg-gray-500";
  const pStartPct = Math.round((player.p_start ?? 0) * 100);

  return (
    <div
      className={clsx(
        "card relative flex flex-col gap-2 hover:border-fpl-green/40 transition-colors cursor-pointer",
        highlight === "captain"      && "border-fpl-green/60 bg-fpl-green/5",
        highlight === "transfer-in"  && "border-cyan-500/50 bg-cyan-500/5",
        highlight === "transfer-out" && "border-red-500/50 bg-red-500/5",
      )}
    >
      {showCaptainStar && (
        <Star size={12} className="absolute top-3 right-3 text-fpl-green fill-fpl-green" />
      )}

      {/* Position badge */}
      <span className={clsx(
        "self-start text-[10px] font-bold px-1.5 py-0.5 rounded uppercase tracking-wider",
        `badge-pos-${player.position}`
      )}>
        {player.position}
      </span>

      {/* Name + status */}
      <div className="flex items-center gap-1.5">
        <span className={clsx("w-2 h-2 rounded-full flex-shrink-0", dotColor)} />
        <p className="text-sm font-semibold text-white leading-tight truncate">
          {player.name}
        </p>
      </div>

      <p className="text-xs text-gray-500 truncate -mt-1">{player.team_name}</p>

      {/* Stats row */}
      <div className="grid grid-cols-2 gap-x-2 gap-y-1 mt-1">
        <Stat label="xP"    value={player.xp?.toFixed(1) ?? "—"} accent />
        <Stat label="£"     value={`${((player.price ?? 0) / 10).toFixed(1)}m`} />
        <Stat label="Start" value={`${pStartPct}%`} />
        <Stat label="Own%"  value={`${(player.ownership_pct ?? 0).toFixed(0)}%`} />
      </div>
    </div>
  );
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div>
      <p className="text-[10px] text-gray-600 uppercase tracking-wider">{label}</p>
      <p className={clsx("text-sm font-semibold", accent ? "text-fpl-green" : "text-gray-300")}>
        {value}
      </p>
    </div>
  );
}
