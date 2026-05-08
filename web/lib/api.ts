/**
 * API client — routes requests to the Python FastAPI backend.
 *
 * In dev:  hits localhost:8000 directly
 * In prod: hits the Railway URL via NEXT_PUBLIC_API_URL env var
 *
 * Set NEXT_PUBLIC_API_URL in Vercel project settings to your Railway URL,
 * e.g. https://fpl-engine-production.up.railway.app
 */

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ??
  (typeof window !== "undefined" && window.location.hostname === "localhost"
    ? "http://localhost:8000"
    : "/api/py");

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${API_URL}${path}`;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${path} failed (${res.status}): ${body}`);
  }
  return res.json() as Promise<T>;
}

// ── Types ─────────────────────────────────────────────────────────────────────

export interface Player {
  element_id: number;
  name: string;
  position: "GKP" | "DEF" | "MID" | "FWD";
  team_name: string;
  price: number;
  xp: number;
  p_start: number;
  p_sub: number;
  p_bench: number;
  e_pts_start: number;
  e_pts_sub: number;
  ownership_pct: number;
  status: string;
}

export interface SquadResult {
  total_xp: number;
  differential_score: number;
  captain_id: number;
  vice_captain_id: number;
  squad: Player[];
  starting_xi: Player[];
  bench: Player[];
  budget_used: number;
  bank: number;
}

export interface Transfer {
  out_id: number;
  out_name: string;
  in_id: number;
  in_name: string;
  xp_gain: number;
  cost_delta: number;
  hit: boolean;
}

export interface CalendarData {
  doubles: Record<string, number[]>;
  blanks: Record<string, number[]>;
  chip_timing: Record<string, Record<string, number>>;
  predicted_events: {
    gameweek: number;
    event_type: string;
    teams: number[];
    confidence: number;
    reason: string;
  }[];
}

export interface BacktestResult {
  strategy: string;
  total_points: number;
  avg_gw_points: number;
  avg_spearman: number | null;
  gw_results: {
    gameweek: number;
    points_scored: number;
    cumulative_points: number;
    captain_points: number;
  }[];
}

export interface RivalReport {
  my_points: number;
  gap_to_leader: number;
  avg_gap: number;
  strategy: string;
  suggested_gamestate: string;
  gws_remaining: number;
  rivals: { name: string; total_points: number; rank: number }[];
  differentials: Player[];
  risk_players: Player[];
}

export interface HealthStatus {
  status: string;
  models_trained: boolean;
  predictions_available: boolean;
  current_gw: number;
  n_players: number;
}

// ── API calls ─────────────────────────────────────────────────────────────────

export const api = {
  // Health
  health: () => apiFetch<HealthStatus>("/health"),

  // Predictions
  predictions: (limit = 50) =>
    apiFetch<Player[]>(`/predictions?limit=${limit}`),

  // Optimization
  optimizeSquad: (body: {
    budget?: number;
    gamestate?: string;
    chip?: string;
    must_include?: number[];
    must_exclude?: number[];
  }) =>
    apiFetch<SquadResult>("/optimize", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  optimizeTransfers: (body: {
    squad_ids: number[];
    free_transfers?: number;
    bank?: number;
    horizon?: number;
  }) =>
    apiFetch<Transfer[]>("/optimize/transfers", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  // Calendar
  calendar: () => apiFetch<CalendarData>("/calendar"),

  // Backtest
  backtest: (body: { strategy?: string; start_gw?: number; end_gw?: number }) =>
    apiFetch<BacktestResult>("/backtest", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  // Rivals
  rivals: (body: { manager_id: number; league_id: number; gameweek: number }) =>
    apiFetch<RivalReport>("/rivals", {
      method: "POST",
      body: JSON.stringify(body),
    }),
};
