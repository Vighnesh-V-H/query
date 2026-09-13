import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Query — Support Agent Demo",
  description:
    "Minimal chat UI over a mocked support-agent pipeline (intent, reply, evidence, decision).",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
