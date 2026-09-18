import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PAIMANA — Infrastructure Early Warning",
  description:
    "Predictive early-warning and Capital-at-Risk monitoring for central sector infrastructure projects (MoSPI, SIH26103).",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  // No webfont is loaded: `system-ui` resolves to SF Pro on Apple hardware and
  // to a native face elsewhere, so typography can never fail to load and take
  // the layout down with it.
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full bg-bg text-fg">{children}</body>
    </html>
  );
}
