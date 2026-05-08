/**
 * GET /api/predictions
 * Reads predictions.csv from the Python engine output and serves it as JSON.
 * The Python engine writes to data/predictions.csv; this route reads and paginates it.
 */

import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";

// CSV → array of objects
function parseCSV(csv: string): Record<string, string>[] {
  const lines = csv.trim().split("\n");
  if (lines.length < 2) return [];
  const headers = lines[0].split(",").map((h) => h.trim());
  return lines.slice(1).map((line) => {
    const values = line.split(",");
    const row: Record<string, string> = {};
    headers.forEach((h, i) => {
      row[h] = (values[i] ?? "").trim();
    });
    return row;
  });
}

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const position = searchParams.get("position") ?? "all";
  const limit = parseInt(searchParams.get("limit") ?? "50", 10);
  const offset = parseInt(searchParams.get("offset") ?? "0", 10);

  // Look for predictions file relative to project root (one level above /web)
  const dataPath = path.join(process.cwd(), "..", "data", "predictions.csv");

  if (!fs.existsSync(dataPath)) {
    return NextResponse.json(
      { error: "Predictions not yet generated. Run the Python engine first." },
      { status: 404 }
    );
  }

  const csv = fs.readFileSync(dataPath, "utf-8");
  let rows = parseCSV(csv);

  // Filter by position
  if (position !== "all") {
    rows = rows.filter((r) => r.position === position.toUpperCase());
  }

  // Sort by xp descending
  rows.sort((a, b) => parseFloat(b.xp ?? "0") - parseFloat(a.xp ?? "0"));

  const total = rows.length;
  const paginated = rows.slice(offset, offset + limit);

  return NextResponse.json({
    total,
    offset,
    limit,
    predictions: paginated,
  });
}
