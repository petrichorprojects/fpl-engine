import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow calling the local Python engine via rewrites in dev
  async rewrites() {
    return [
      {
        source: "/python/:path*",
        destination: "http://localhost:8000/:path*",
      },
    ];
  },
};

export default nextConfig;
