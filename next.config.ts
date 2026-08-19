import type { NextConfig } from "next";

/**
 * On Vercel the Python function in api/ is served at the same origin, and vercel.json
 * routes /api/* to it -- no rewrite is needed or wanted here.
 *
 * Locally the FastAPI app runs as a separate process on another port, so /api/* is
 * proxied to it. The previous config applied that localhost proxy unconditionally, which
 * would have pointed production traffic at 127.0.0.1:8000.
 */
const LOCAL_API = process.env.LOCAL_API_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    if (process.env.VERCEL) return [];
    return [{ source: "/api/:path*", destination: `${LOCAL_API}/api/:path*` }];
  },

  // Surfaced in the UI footer and useful when diagnosing a stale deployment.
  env: {
    NEXT_PUBLIC_BUILD_SHA: process.env.VERCEL_GIT_COMMIT_SHA?.slice(0, 7) ?? "local",
  },
};

export default nextConfig;
