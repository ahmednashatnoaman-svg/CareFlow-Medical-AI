import type { Metadata, Viewport } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans, Space_Grotesk } from "next/font/google";
import "./globals.css";

/*
 * Type pairing for "Clinical Instrumentation":
 *   Space Grotesk — technical grotesque with real character; carries the display voice
 *     without the neutrality of Inter/Roboto.
 *   IBM Plex Sans — institutional heritage, designed for dense technical documents;
 *     the right register for clinical copy.
 *   IBM Plex Mono — every measured value renders here, so numerics stay tabular and
 *     visually distinct from prose.
 */
const display = Space_Grotesk({
  variable: "--font-display",
  subsets: ["latin"],
  weight: ["500", "700"],
  display: "swap",
});

const body = IBM_Plex_Sans({
  variable: "--font-body",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  display: "swap",
});

const mono = IBM_Plex_Mono({
  variable: "--font-mono-plex",
  subsets: ["latin"],
  weight: ["400", "500"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "CareFlow — Dual-Mode Clinical AI",
  description:
    "Graph RAG diagnostic triage and WHO guideline retrieval, with live interview telemetry and cited clinical evidence.",
  applicationName: "CareFlow",
  authors: [{ name: "CareFlow" }],
  openGraph: {
    title: "CareFlow — Dual-Mode Clinical AI",
    description:
      "Graph RAG diagnostic triage and WHO guideline retrieval with cited evidence.",
    type: "website",
  },
};

export const viewport: Viewport = {
  themeColor: "#0b0f14",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={`${display.variable} ${body.variable} ${mono.variable} h-full`}
      suppressHydrationWarning
    >
      <body className="min-h-full antialiased">{children}</body>
    </html>
  );
}
