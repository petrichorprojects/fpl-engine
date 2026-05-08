import { NextResponse } from "next/server";

const PYTHON_API = process.env.PYTHON_API_URL ?? "http://localhost:8000";

export async function GET() {
  try {
    const res = await fetch(`${PYTHON_API}/calendar`, { cache: "no-store" });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (e) {
    return NextResponse.json({ error: "Engine unavailable", detail: String(e) }, { status: 503 });
  }
}
