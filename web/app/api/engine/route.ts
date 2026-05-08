/**
 * POST /api/engine
 * Spawns the Python engine as a child process and streams progress back.
 * Body: { gamestate, budget, chip }
 *
 * In production on Vercel: engine runs as a separate service / cron job
 * and writes output to shared storage. This route triggers via webhook.
 *
 * In Electron/local: runs the Python process directly.
 */

import { NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";
import path from "path";

export const maxDuration = 300; // 5 min timeout for Vercel

export async function POST(req: NextRequest) {
  const body = await req.json().catch(() => ({}));
  const { gamestate = "neutral", budget = 1000, chip = "none" } = body;

  const projectRoot = path.join(process.cwd(), "..");
  const scriptPath = path.join(projectRoot, "run_engine.py");
  const uvPath = process.env.UV_PATH ?? "uv";

  // Build args
  const args = [
    "run",
    "python",
    scriptPath,
    "--skip-fetch",
    "--gamestate", gamestate,
    "--budget", String(budget),
    "--chip", chip,
  ];

  return new Promise<NextResponse>((resolve) => {
    const proc = spawn(uvPath, args, {
      cwd: projectRoot,
      env: { ...process.env },
    });

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (d: Buffer) => { stdout += d.toString(); });
    proc.stderr.on("data", (d: Buffer) => { stderr += d.toString(); });

    proc.on("close", (code) => {
      if (code === 0) {
        resolve(NextResponse.json({ success: true, output: stdout }));
      } else {
        resolve(
          NextResponse.json(
            { success: false, error: stderr || "Engine failed", output: stdout },
            { status: 500 }
          )
        );
      }
    });

    proc.on("error", (err) => {
      resolve(
        NextResponse.json(
          { success: false, error: err.message },
          { status: 500 }
        )
      );
    });
  });
}
